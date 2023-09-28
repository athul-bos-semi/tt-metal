// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/program.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/reports/compilation_reporter.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "common/executor.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/detail/kernel_cache.hpp"

#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tools/profiler/profiler.hpp"
#include "llrt/tt_debug_print_server.hpp"

namespace tt::tt_metal {


namespace{
    std::atomic<bool> enable_persistent_kernel_cache = false;

    void GenerateBinaries(Device *device, build_kernel_for_riscv_options_t *build_options, const std::string &op_path_suffix, Kernel *kernel) {
        ZoneScoped;
        const std::string tracyPrefix = "GenerateBinaries_";
        ZoneName( (tracyPrefix + op_path_suffix).c_str(), op_path_suffix.length() + tracyPrefix.length());
        try {
            generate_descriptors(build_options, op_path_suffix);
            kernel->generate_binaries(device, build_options, op_path_suffix);
        } catch (std::runtime_error &ex) {
            log_fatal("Failed to generate binaries for {} {}", kernel->name(), ex.what());
        }
    }


    #ifdef GENERATE_HASH_LOG
    #include <fstream>
    #endif

    size_t KernelCompileHash(
        Kernel *kernel, build_kernel_for_riscv_options_t &build_options, const int &device_id) {
        string compile_hash_str = std::to_string(std::hash<tt_hlk_desc>{}(build_options.hlk_desc));
        compile_hash_str += kernel->compute_hash();
        size_t compile_hash = std::hash<std::string>{}(compile_hash_str);

    #ifdef GENERATE_HASH_LOG
        static std::ofstream f("/tmp/hashlog.txt");
        static std::mutex mutex_;
        {
            unique_lock<mutex> lock;
            f << kernel->name() << " :: "
            << std::hash<tt_hlk_desc>{}(build_options.hlk_desc) << " :: "
            << kernel->compute_hash() << " :: "
            << compile_hash_str << " "
            << compile_hash << std::endl << std::flush;
        }
    #endif
        return compile_hash;
    }

}
namespace detail{
    void EnablePersistentKernelCache()
    {
        enable_persistent_kernel_cache = true;
    }

    void DisablePersistentKernelCache()
    {
        enable_persistent_kernel_cache = false;
    }

    inline void CompileBlankKernel(Device *device) {
        // Crude way to check if blank_op needs to be compiled or not
        // TODO(pgk):
        //  - fw is compiled every run
        //  - for unknown reasons, fw size can vary run to run
        //  - kernels from one run linked against fw from another run may clash
        //  - rebuid blank kernels once per run
        static bool compiled = false;
        if (compiled) {
            return;
        }

        build_kernel_for_riscv_options_t blank_build_options(device->id(), "blank_op");
        struct hlk_args_t {
            std::int32_t dummy;
        };
        void *hlk_args = new hlk_args_t{
            .dummy = 0,
        };
        blank_build_options.set_hlk_args_all_cores(hlk_args, sizeof(hlk_args_t));
        blank_build_options.set_hlk_file_name_all_cores("tt_metal/kernels/compute/blank.cpp");
        blank_build_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
        blank_build_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
        std::string arch_name = tt::get_string_lowercase(device->arch());

        generate_binaries_params_t default_params;
        detail::GenerateDeviceHeaders(device, &blank_build_options, blank_build_options.name);
        generate_binaries_all_riscs(&blank_build_options, blank_build_options.name, arch_name, default_params);

        compiled = true;
    }
}
auto Program::semaphores_on_core(const CoreCoord &core) const {
    std::vector<std::reference_wrapper<const Semaphore>> semaphores;
    for ( const Semaphore & s : this->semaphores_) {
        if (s.initialized_on_logical_core(core)) {
            semaphores.emplace_back(std::cref(s));
        }
    }
    return semaphores;
}

std::atomic<u64> Program::program_counter = 0;

Program::Program(): id(program_counter++),worker_crs_({}), compile_needed_(false), circular_buffer_allocation_needed_(false) {}

void Program::add_kernel(Kernel *kernel) {
    this->invalidate_compile();
    kernel_ids_.push_back(kernel->id());
    core_to_kernel_group_.clear();
    kernel_by_id_[kernel->id()] = kernel;
}

Kernel *Program::get_kernel(KernelID kernel_id) const {
    TT_ASSERT(this->kernel_by_id_.find(kernel_id) != this->kernel_by_id_.end(), "Expected Kernel with ID {} to be in Program {}", kernel_id, this->id);
    return this->kernel_by_id_.at(kernel_id);
}

KernelGroup::KernelGroup() {
    launch_msg.run = RUN_MSG_GO;
}

void KernelGroup::update(const Kernel *kernel) {
    RISCV riscv_processor = kernel->processor();
    switch (riscv_processor) {
        case RISCV::BRISC:
            this->riscv0_id = kernel->id();
            this->launch_msg.enable_brisc = true;
            break;
        case RISCV::NCRISC:
            this->riscv1_id = kernel->id();
            this->launch_msg.enable_ncrisc = true;
            break;
        case RISCV::COMPUTE:
            this->compute_id = kernel->id();
            this->launch_msg.enable_triscs = true;
            break;
        default:
            TT_ASSERT(false, "Unsupported kernel processor!");
    }
}

KernelGroup * Program::kernels_on_core(const CoreCoord &core) {
    std::map<CoreCoord, KernelGroup>& kernel_groups = core_to_kernel_group();
    auto result = kernel_groups.find(core);
    return (result == kernel_groups.end()) ? nullptr : &result->second;
}

std::map<CoreCoord, KernelGroup>& Program::core_to_kernel_group() {
    if (core_to_kernel_group_.size() == 0) {
        for (auto &[kernel_id, kernel] : this->kernel_by_id_) {
            for (auto core : kernel->logical_cores()) {
                KernelGroup &kernel_group = core_to_kernel_group_[core];
                kernel_group.update(kernel);
            }
        }
    }

    return core_to_kernel_group_;
}

std::vector<std::string> Program::cores_to_ops() const {
    std::vector<std::string> ops;

    for (const auto &core : this->logical_cores()) {
        for (auto kernel_id : this->kernel_ids_) {
        auto kernel = this->get_kernel(kernel_id);
        auto cores = kernel->logical_cores();
            if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
                ops.push_back(kernel->name());
            }
        }
    }
    return ops;
}

void Program::CircularBufferAllocator::add_index(u32 index) {
    if (index > NUM_CIRCULAR_BUFFERS) {
        log_fatal(tt::LogMetal, "Invalid circular buffer index: {} should be between 0 and {}", index, NUM_CIRCULAR_BUFFERS);
    }
    if (this->indices.to_ulong() & (1 << index)) {
        log_fatal(tt::LogMetal, "Invalid circular buffer index: Cannot add circular buffer at index {}, another circular buffer already exists", index);
    }
    this->indices[index] = 1;
}

// CBs on a core are sequential so the next available address for a local buffer is the end of the last
u64 Program::CircularBufferAllocator::get_address_candidate() const {
    return this->l1_regions.back().second;
}

void Program::CircularBufferAllocator::mark_address(u64 address, u64 size) {
    auto &last_region = this->l1_regions.back();
    log_assert(address >= last_region.second, "Local buffer address {} has to append to last L1 region [{}, {}) or be at a higher address", address, last_region.first, last_region.second);
    if (address == last_region.second) {
        last_region.second += size;
    } else {
        this->l1_regions.push_back({address, address + size});
    }
}

CircularBufferID Program::add_circular_buffer(const CoreRangeSet &core_range_set, const CircularBufferConfig &config) {
    this->invalidate_compile();
    this->invalidate_circular_buffer_allocation();
    std::shared_ptr<CircularBuffer> circular_buffer = std::make_shared<CircularBuffer>(core_range_set, config);

    // Mark which buffer indices are being used on each core the circular buffer is used on
    for (const auto &core_range : core_range_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord logical_core(x, y);
                auto &cb_config = this->per_core_cb_allocator_[logical_core];

                for (auto buffer_index : circular_buffer->buffer_indices()) {
                    cb_config.add_index(buffer_index);
                }
            }
    }
    }

    this->circular_buffers_.push_back(circular_buffer);
    this->circular_buffer_by_id_.insert({circular_buffer->id(), circular_buffer});
    return circular_buffer->id();
}

std::shared_ptr<CircularBuffer> Program::get_circular_buffer(CircularBufferID cb_id) const {
    if (this->circular_buffer_by_id_.find(cb_id) == this->circular_buffer_by_id_.end()) {
        log_fatal(tt::LogMetal, "No circular buffer with id {} exists in Program {}", cb_id, this->id);
    }
    return this->circular_buffer_by_id_.at(cb_id);
}

const std::vector<std::shared_ptr<CircularBuffer>> Program::circular_buffers_on_core(const CoreCoord &core) const {
    std::vector<std::shared_ptr<CircularBuffer>> cbs_on_core;
    for (auto circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_core(core)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

const std::vector<std::shared_ptr<CircularBuffer>> Program::circular_buffers_on_corerange(const CoreRange & cr) const {
    std::vector<std::shared_ptr<CircularBuffer>> cbs_on_core;
    for (auto circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_corerange(cr)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

void Program::invalidate_circular_buffer_allocation() {
    if (this->circular_buffer_allocation_needed_) {
        return;
    }
    for (auto &[logical_core, cb_allocator] : this->per_core_cb_allocator_) {
        cb_allocator.reset_available_addresses();
    }
    this->circular_buffer_allocation_needed_ = true;
}

void Program::allocate_circular_buffers() {
    if (not this->circular_buffer_allocation_needed_) {
        return;
    }

    for (std::shared_ptr<CircularBuffer> circular_buffer : this->circular_buffers_) {
        std::optional<uint64_t> computed_addr = std::nullopt;
        std::vector<std::reference_wrapper<CircularBufferAllocator>> cb_allocators;
        for (const CoreRange &core_range : circular_buffer->core_ranges().ranges()) {
            for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                    CoreCoord logical_core(x, y);
                    auto &cb_allocator = this->per_core_cb_allocator_.at(logical_core);

                    // Need the max available address across all cores circular buffer is placed on
                    auto candidate_addr = cb_allocator.get_address_candidate();
                    if (not computed_addr.has_value()) {
                        computed_addr = candidate_addr;
                    } else {
                        computed_addr = std::max(computed_addr.value(), candidate_addr);
                    }

                    cb_allocators.push_back(cb_allocator);
                }
            }
        }

        // okay to access config and invalidate circular buffer address because it will be set below
        std::optional<uint32_t> requested_address = circular_buffer->config().requested_address();
        if (requested_address.has_value()) {
            if (requested_address.value() < computed_addr.value()) {
                log_fatal(tt::LogMetal, "Specified address {} should be at max local buffer region for core range set, try {} instead", requested_address.value(), computed_addr.value());
            }
            computed_addr = requested_address;
        }

        for (auto &cb_allocator : cb_allocators) {
            cb_allocator.get().mark_address(computed_addr.value(), circular_buffer->size());
        }

        circular_buffer->set_address(computed_addr.value());
    }
    this->circular_buffer_allocation_needed_ = false;
}

void Program::validate_circular_buffer_region(const Device *device, std::optional<CoreCoord> logical_core) const {
    auto highest_cb_l1_region = [&](const CoreCoord &core) {
        if (this->per_core_cb_allocator_.find(core) == this->per_core_cb_allocator_.end()) {
            return std::make_pair((u64)L1_UNRESERVED_BASE, (u64)L1_UNRESERVED_BASE);
        }
        return this->per_core_cb_allocator_.at(core).l1_regions.back();
    };

    auto validate_cb_space_and_l1_buffer_space_disjoint = [&](const CoreCoord &core, const std::pair<u64, u64> &cb_space) {
        if (cb_space.second > device->l1_size()) {
            log_fatal(tt::LogMetal, "Local buffers on core {} grow to {} B which is beyond max L1 size of {} B", core.str(), cb_space.second, device->l1_size());
        }

        auto bank_ids = device->bank_ids_from_logical_core(core);
        if (bank_ids.size() != 1) {
            log_fatal(tt::LogMetal, "Expected one bank on core that holds local and L1 buffers but logical core {} has {} banks", core.str(), bank_ids.size());
        }

        auto lowest_address = allocator::lowest_occupied_l1_address(*device->allocator_, bank_ids.at(0));
        if (lowest_address.has_value()) {
            if (lowest_address.value() < cb_space.second) {
                log_fatal(tt::LogMetal, "Circular buffers in program {} clash with L1 buffers on core {}. L1 buffer allocated at {} and local buffers end at {}", this->id, core.str(), lowest_address.value(), cb_space.second);
            }
        }
    };

    if (logical_core.has_value()) {
        const auto &cb_space = highest_cb_l1_region(logical_core.value());
        validate_cb_space_and_l1_buffer_space_disjoint(logical_core.value(), cb_space);
    } else {
        for (const auto &[core, cb_config] : this->per_core_cb_allocator_) {
            const auto &cb_space = highest_cb_l1_region(core);
            validate_cb_space_and_l1_buffer_space_disjoint(core, cb_space);
        }
    }
}

size_t Program::num_semaphores(const CoreCoord &core) const {
    return semaphores_on_core(core).size();
}

size_t Program::num_semaphores() const {
    return semaphores_.size();
}

uint32_t Program::semaphore_address ( uint32_t sem_idx ) const {
    return semaphores_.at(sem_idx).address();
}

void Program::init_semaphores( const Device & device, const CoreCoord &logical_core ) const{
    auto semaphores_on_core = this->semaphores_on_core(logical_core);
    for (auto semaphore : semaphores_on_core) {
        llrt::write_hex_vec_to_core(device.cluster(), device.id(), device.worker_core_from_logical_core(logical_core), {semaphore.get().initial_value()}, semaphore.get().address());
    }
}

void Program::add_semaphore(const CoreRangeSet & crs, uint32_t address, uint32_t init_value) {
    this->invalidate_compile();
    semaphores_.emplace_back(Semaphore( crs, address, init_value));
}

std::vector<CoreCoord> Program::logical_cores() const {
    std::vector<CoreCoord> cores_in_program;
    std::set<CoreCoord> unique_cores;
    for (const auto &[kernel_id, kernel] : this->kernel_by_id_) {
        for (auto core : kernel->logical_cores()) {
            if (unique_cores.find(core) != unique_cores.end()) {
                continue;
            }
            unique_cores.insert(core);
            cores_in_program.push_back(core);
        }
    }
    return cores_in_program;
}

void Program::construct_core_range_set_for_worker_cores() {
    bool found_kernels = false;
    for (const auto &[kernel_id, kernel] : this->kernel_by_id_) {
        this->worker_crs_.merge ( kernel->core_range_set());
        found_kernels = true;
    }
    TT_ASSERT(!found_kernels || this->worker_crs_.ranges().size() >= 1, "Invalid core range set");
}


// Just adds blank kernels, doesn't compile them or read binaries
void Program::add_blank_kernels(Device *device) {
    ZoneScoped;

    // This can be smarter by combining core ranges into maximal rectangles but this code can be removed once we load BRISC FW separately from the kernel binary
    std::set<CoreRange> unique_core_ranges_without_brisc_kernel,
                        unique_core_ranges_without_ncrisc_kernel,
                        unique_core_ranges_without_compute_kernel ;
    vector<KernelID> blank_ids;
    for (auto &[logical_core, kernel_group] : this->core_to_kernel_group()) {
        CoreRange core_range = {.start = logical_core, .end = logical_core};
        if (not kernel_group.riscv0_id.has_value())
            unique_core_ranges_without_brisc_kernel.insert(core_range);
        if (not kernel_group.riscv1_id.has_value())
            unique_core_ranges_without_ncrisc_kernel.insert(core_range);
        if (not kernel_group.compute_id.has_value())
            unique_core_ranges_without_compute_kernel.insert(core_range);
    }

    if (not unique_core_ranges_without_brisc_kernel.empty()) {
        CoreRangeSet core_range_set = CoreRangeSet(unique_core_ranges_without_brisc_kernel);
        KernelID blank_kernel_id = CreateDataMovementKernel(
            *this, "tt_metal/kernels/dataflow/blank.cpp", core_range_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        blank_ids.push_back(blank_kernel_id);
    }

    if (not unique_core_ranges_without_ncrisc_kernel.empty()) {
        CoreRangeSet core_range_set = CoreRangeSet(unique_core_ranges_without_ncrisc_kernel);
        KernelID blank_kernel_id = CreateDataMovementKernel(
            *this, "tt_metal/kernels/dataflow/blank.cpp", core_range_set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        blank_ids.push_back(blank_kernel_id);
    }

    if (not unique_core_ranges_without_compute_kernel.empty()) {
        CoreRangeSet core_range_set = CoreRangeSet(unique_core_ranges_without_compute_kernel);
        KernelID blank_kernel_id = CreateComputeKernel(*this, "tt_metal/kernels/compute/blank.cpp", core_range_set);
        blank_ids.push_back(blank_kernel_id);
    }
}

void Program::set_cb_data_fmt(
    Device *device, Kernel *kernel, build_kernel_for_riscv_options_t &build_options) const {
    ZoneScoped;
    for (auto logical_cr : kernel->logical_coreranges()) {
        auto cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (auto circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                build_options.set_cb_dataformat_all_cores(static_cast<CB>(buffer_index), circular_buffer->data_format(buffer_index));
            }
        }
    }
}

void Program::compile( Device * device )
{
    if( !compile_needed_) return;

    TT_ASSERT(
        device->is_initialized(),
        "Device needs to be initialized before program {} compilation! Generating headers for banking information is "
        "dependent on information that is set during device initialization.",
        this->get_id());

    tt::tt_metal::detail::CompileBlankKernel(device);

    detail::ProfileTTMetalScope profile_this = detail::ProfileTTMetalScope("CompileProgram");
    bool profile_kernel = getDeviceProfilerState();
    std::vector<std::future<void>> events;
    log_assert(
        !(profile_kernel && tt_is_print_server_running()), "Debug print server is running, profiling is not allowed");
    tt_set_profiler_state_for_debug_print(profile_kernel);

    // add blank kernels to program, we do this serially before we start compiling all kernels in parallel
    this->add_blank_kernels(device);

    // compile all kernels in parallel, including the blanks
    for (auto kernel_id : this->kernel_ids()) {
        auto kernel = this->get_kernel(kernel_id);
        events.emplace_back ( detail::async ( [kernel, device, this] {
            build_kernel_for_riscv_options_t build_options(device->id(), kernel->name());
            ZoneScoped;

            kernel->set_build_options(build_options);
            this->set_cb_data_fmt(device, kernel, build_options);

            auto kernel_hash = KernelCompileHash(kernel, build_options, device->id());
            std::string kernel_path_suffix = kernel->name() + "/" + std::to_string(kernel_hash);

            bool cache_hit = true;
            bool path_exists = std::filesystem::exists(build_options.outpath + kernel_path_suffix);
            if ( enable_persistent_kernel_cache && path_exists ) {
                if ( not detail::HashLookup::inst().exists(kernel_hash) ) detail::HashLookup::inst().add(kernel_hash);
            } else if ( detail::HashLookup::inst().add(kernel_hash) ) {
                cache_hit = false;
                GenerateBinaries(device, &build_options, kernel_path_suffix, kernel);
            }
            if (detail::CompilationReporter::enabled()) {
                detail::CompilationReporter::inst().add_kernel_compile_stats(*this, kernel, cache_hit, kernel_hash);
            }

            kernel->set_binary_path(kernel_path_suffix);
        } ) );
    }

    for (auto & f : events)
        f.wait();

    for (auto kernel_id : this->kernel_ids()) {
        auto kernel = this->get_kernel(kernel_id);
        events.emplace_back ( detail::async ( [kernel, device] { kernel->read_binaries(device->id()); }));
    }

    for (auto & f : events)
        f.wait();

    this->construct_core_range_set_for_worker_cores();

    if (detail::CompilationReporter::enabled()) {
        detail::CompilationReporter::inst().flush_program_entry(*this, enable_persistent_kernel_cache);
    }
    if (detail::MemoryReporter::enabled()) {
        detail::MemoryReporter::inst().flush_program_memory_usage(*this, device);
    }
    compile_needed_ = false;
}

Program::~Program() {
    for (const auto &[kernel_id, kernel] : this->kernel_by_id_) {
        delete kernel;
    }
}

}  // namespace tt::tt_metal
