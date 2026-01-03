#include "context.h"
#include "swapchain.h"
#include "texture.h"
#include "texture_manager.h"
#include "model.h"
#include "model_manager.h"
#include "vulkan_utils.h"
#include "ray.h"
#include "mouse_interactor.h"
#include "particle_manager.h"
#include "gpu_profiler.h"

#define VRDX_IMPLEMENTATION
#include "cloth_xpbd_pass_gpu.h"

ClothXpbdPassGPU::ClothXpbdPassGPU(Context& context, Swapchain& swapchain, ParticleManager& particleManager, ModelManager& modelManager)
	: context_(context), particle_manager_(particleManager), model_manager_(modelManager)
{
	CreateCommandBuffers();
	CreateDescriptorSetLayout();
	CreateDescriptorPools();
	CreateUniformBuffers();

	//std::cout << std::filesystem::current_path() << std::endl;

	sim_datas_.total_particles_ = particleManager.total_particles_;
	sim_datas_.total_indices_ = particleManager.total_indices_;
	sim_datas_.total_tri_ = particleManager.total_tries_;

	CreateClothConstraintDatas();
	CreateSoftBodyConstraintDatas();
	CreateColiiders();
	CreateSSBOBuffers();
	CreateDescriptorSets();
	CreateComputePipelineLayouts();
	CreateComputePipelines();
	CreateVrdxSorter();

	gpu_profiler_ = std::make_unique<GpuProfiler>(context, sim_datas_);
}

ClothXpbdPassGPU::~ClothXpbdPassGPU()
{
	if (radix_.sorter) {
		vrdxDestroySorter(radix_.sorter);
	}
}

void ClothXpbdPassGPU::UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize)
{
	if (mouseInteractor.is_left_button_down_event)
	{
		sim_datas_.push_constants_.mouse_interact.select_mode = 1;

		Ray ray = mouseInteractor.CalculateMouseRay(camera, viewportSize);

		sim_datas_.push_constants_.mouse_interact.ray_origin = ray.origin;
		sim_datas_.push_constants_.mouse_interact.ray_dir = ray.direction;

	}
	else if (!mouseInteractor.is_left_button_down_event && mouseInteractor.is_left_down)
	{
		sim_datas_.push_constants_.mouse_interact.select_mode = 2;

		Ray ray = mouseInteractor.CalculateMouseRay(camera, viewportSize);

		sim_datas_.push_constants_.mouse_interact.ray_origin = ray.origin;
		sim_datas_.push_constants_.mouse_interact.ray_dir = ray.direction;
	}
	else
		sim_datas_.push_constants_.mouse_interact.select_mode = 0;

	if (mouseInteractor.depth_state == vku::DepthState::MOUSE_DEPTH_IN)
		sim_datas_.push_constants_.mouse_interact.depth_mode = vku::DepthState::MOUSE_DEPTH_IN;
	else if (mouseInteractor.depth_state == vku::DepthState::MOUSE_DEPTH_OUT)
		sim_datas_.push_constants_.mouse_interact.depth_mode = vku::DepthState::MOUSE_DEPTH_OUT;
	else
		sim_datas_.push_constants_.mouse_interact.depth_mode = vku::DepthState::MOUSE_DEPTH_NONE;
	mouseInteractor.depth_state = vku::DepthState::MOUSE_DEPTH_NONE;
}

void ClothXpbdPassGPU::UpdateComputeUBO(uint32_t currentFrame, ModelManager& modelManager)
{
	ubo_.datas.sim_params.dt = 1.0f / sim_datas_.frame_dt / sim_datas_.substeps;

	ubo_.datas.sim_params.num_particles = sim_datas_.total_particles_;
	ubo_.datas.sim_params.num_edges = sim_datas_.num_edges;
	ubo_.datas.sim_params.num_shears = sim_datas_.num_shears;
	ubo_.datas.sim_params.num_bends = sim_datas_.num_bends;
	ubo_.datas.sim_params.num_areas = sim_datas_.num_areas;
	ubo_.datas.sim_params.num_tries = sim_datas_.total_tri_;
	ubo_.datas.sim_params.num_volumes = sim_datas_.num_volumes;
	ubo_.datas.sim_params.num_colliders = sim_datas_.num_colliders;

	float dt = ubo_.datas.sim_params.dt;
	float r = ubo_.datas.sim_params.collision_radius;
	float k = 4.0f;
	ubo_.datas.sim_params.max_speed = k * r / dt;

	const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * ubo_.size.sim_params);
	auto* dst = static_cast<std::byte*>(ubo_.mapped.sim_params) + baseOffset;

	std::memcpy(dst, &ubo_.datas.sim_params, sizeof(SimUBO::Data::SimParams));
}

void ClothXpbdPassGPU::RecordCompute(uint32_t currentFrame, vku::TestScene& testScene)
{
	auto& cmd = cmds_[currentFrame];
	auto& pmSSBO = particle_manager_.ssbos_;
	auto& dSSBO = sim_datas_.ssbos_;
	auto& pm = particle_manager_;
	auto& gp = *gpu_profiler_;

	cmd.reset();
	cmd.begin({});

	gp.timestamp_steps_ = 0;
	const auto stage = vk::PipelineStageFlagBits2::eComputeShader;
	auto TS = [&](uint32_t& idx) {
		cmd.writeTimestamp2(stage, *gp.timestamp_pool_, idx++);
		};

	cmd.resetQueryPool(*gp.timestamp_pool_, 0, gp.slots_per_compute_);

	TS(gp.timestamp_steps_); // Start

	ResetTestScene(cmd, testScene);
	CopyColliders(cmd);

	uint32_t simparamOffset = currentFrame * static_cast<uint32_t>(ubo_.size.sim_params);
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eCompute, pipeline_layouts_.common,
		0,
		{ sets_.sim_params, sets_.cloth_compute },
		{ simparamOffset }
	);

	auto CellDiv = [](uint32_t n, uint32_t d) { return (n + d - 1) / d; };

	uint32_t groupsTotal = CellDiv(sim_datas_.total_particles_, 256);
	uint32_t groupsTri = CellDiv((sim_datas_.total_indices_) / 3, 256u);

	for (uint32_t step = 0; step < sim_datas_.substeps; step++)
	{
		// Wind
		TS(gp.timestamp_steps_);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.wind);
		cmd.dispatch(groupsTri, 1, 1);
		TS(gp.timestamp_steps_);

		// Integrate
		TS(gp.timestamp_steps_);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.integrate);
		cmd.pushConstants < SimData::PushConstant > (
			*pipeline_layouts_.common,
			vk::ShaderStageFlagBits::eCompute,
			0,
			sim_datas_.push_constants_);

		cmd.dispatch(groupsTotal, 1, 1);
		TS(gp.timestamp_steps_);
		vku::ssboCompWtoCompRW(cmd, pm.ssbos_.pred_position);

		// Clear Lambdas
		uint32_t Nmax = std::max(
			{
				sim_datas_.total_particles_,
				sim_datas_.num_edges,
				sim_datas_.num_bends,
				sim_datas_.num_shears,
				sim_datas_.num_areas,
				sim_datas_.num_volumes,
				sim_datas_.num_lras
			}
		);
		TS(gp.timestamp_steps_);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.clear_lambdas);
		uint32_t groups = CellDiv(Nmax, 256);
		cmd.dispatch(groups, 1, 1);
		TS(gp.timestamp_steps_);

		// Broad Phase
		if (step % sim_datas_.broadphase_interval_ == 0)
		{
			vku::Barrier2(cmd,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

			// build_hash
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.self_collision)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.build_hash);
				cmd.dispatch(groupsTotal, 1, 1);
			}
			TS(gp.timestamp_steps_);

			vku::BufferBarrier2(
				cmd,
				radix_.storage_buffer,
				0,
				VK_WHOLE_SIZE,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eAllTransfer,
				vk::AccessFlagBits2::eTransferWrite
			);

			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.self_collision)
			{
				vrdxCmdSortKeyValue(
					*cmd, radix_.sorter, sim_datas_.total_particles_,
					*pmSSBO.particle_hash, 0,
					*pmSSBO.sorted_indice, 0,
					*radix_.storage_buffer, 0,
					nullptr, 0
				);
			}
			TS(gp.timestamp_steps_);


			if (sim_datas_.solver_config_.self_collision)
			{
				uint32_t offset = 0;
				uint64_t size = VK_WHOLE_SIZE;

				vku::BufferBarrier2(cmd, pmSSBO.start, offset, particle_manager_.ssbo_size_.start,
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
					vk::PipelineStageFlagBits2::eClear,
					vk::AccessFlagBits2::eTransferWrite);
				vku::BufferBarrier2(cmd, pmSSBO.end, offset, particle_manager_.ssbo_size_.end,
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
					vk::PipelineStageFlagBits2::eClear,
					vk::AccessFlagBits2::eTransferWrite);

				cmd.fillBuffer(pmSSBO.start, offset, size, 0xFFFFFFFF);
				cmd.fillBuffer(pmSSBO.end, offset, size, 0);

				vku::BufferBarrier2(cmd, pmSSBO.start, offset, particle_manager_.ssbo_size_.start,
					vk::PipelineStageFlagBits2::eClear,
					vk::AccessFlagBits2::eTransferWrite,
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

				vku::BufferBarrier2(cmd, pmSSBO.end, offset, particle_manager_.ssbo_size_.end,
					vk::PipelineStageFlagBits2::eClear,
					vk::AccessFlagBits2::eTransferWrite,
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
			}

			uint32_t simparamOffset = currentFrame * static_cast<uint32_t>(ubo_.size.sim_params);
			cmd.bindDescriptorSets(
				vk::PipelineBindPoint::eCompute, pipeline_layouts_.common,
				0,
				{ sets_.sim_params, sets_.cloth_compute },
				{ simparamOffset }
			);

			// build_cell
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.self_collision)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.build_cell);
				cmd.dispatch(groupsTotal, 1, 1);
			}
			TS(gp.timestamp_steps_);

			vku::Barrier2(cmd,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

			// build_neighbor
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.self_collision)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.build_neighbor);
				cmd.dispatch(groupsTotal, 1, 1);
			}
			TS(gp.timestamp_steps_);

			vku::Barrier2(cmd,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
		}

		for (uint32_t iter = 0; iter < sim_datas_.iterations; iter++)
		{
			// Solve Coloring - Stretch
			// TS -> include Barrier Time
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.stretch)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_stretch);

				for (uint32_t c = 0; c < sim_datas_.num_colors; ++c) {
					uint32_t base = sim_datas_.pass_offsets[c];
					uint32_t count = sim_datas_.pass_offsets[c + 1] - sim_datas_.pass_offsets[c];
					if (!count) continue;

					sim_datas_.push_constants_.solve.base = base;
					sim_datas_.push_constants_.solve.count = count;
					sim_datas_.push_constants_.solve.compliance = sim_datas_.compliance.stretch;
					sim_datas_.push_constants_.solve.beta = sim_datas_.beta.stretch;

					cmd.pushConstants<SimData::PushConstant>(
						*pipeline_layouts_.common,
						vk::ShaderStageFlagBits::eCompute,
						0,
						sim_datas_.push_constants_);

					uint32_t groups = (count + 256 - 1) / 256;
					cmd.dispatch(groups, 1, 1);
					vku::ssboCompWtoCompRW(cmd, pmSSBO.pred_position);
				}
			}
			TS(gp.timestamp_steps_);

			// Solve Shear
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.shear)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_shear);

				sim_datas_.push_constants_.solve.compliance = sim_datas_.compliance.shear;
				cmd.pushConstants<SimData::PushConstant>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sim_datas_.push_constants_);

				uint32_t groupsShear = CellDiv(sim_datas_.num_shears, 256);
				cmd.dispatch(groupsShear, 1, 1);
			}
			TS(gp.timestamp_steps_);

			// Solve Bend
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.bend)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_bend);
				uint32_t base = 0;
				uint32_t count = sim_datas_.num_bends;
				sim_datas_.push_constants_.solve.base = base;
				sim_datas_.push_constants_.solve.count = count;
				sim_datas_.push_constants_.solve.compliance = sim_datas_.compliance.bend;
				cmd.pushConstants<SimData::PushConstant>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sim_datas_.push_constants_);

				uint32_t group = (count + 256 - 1) / 256;
				cmd.dispatch(group, 1, 1);
			}
			TS(gp.timestamp_steps_);

			// Solve Area
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.area)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_area);
				uint32_t base = 0;
				uint32_t count = sim_datas_.num_areas;
				sim_datas_.push_constants_.solve.base = base;
				sim_datas_.push_constants_.solve.count = count;
				sim_datas_.push_constants_.solve.compliance = sim_datas_.compliance.area;
				cmd.pushConstants<SimData::PushConstant>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sim_datas_.push_constants_);

				uint32_t group = (count + 256 - 1) / 256;
				cmd.dispatch(group, 1, 1);
			}
			TS(gp.timestamp_steps_);

			// solve_softbody_stretch
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.softbody_stretch)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_softbody_stretch);

				uint32_t base = sim_datas_.pass_offsets[sim_datas_.num_colors];
				uint32_t count = sim_datas_.pass_offsets[sim_datas_.num_colors + 1] - sim_datas_.pass_offsets[sim_datas_.num_colors];
				if (!count) continue;

				sim_datas_.push_constants_.solve.base = base;
				sim_datas_.push_constants_.solve.count = count;
				sim_datas_.push_constants_.solve.compliance = sim_datas_.compliance.softbody_stretch;
				cmd.pushConstants<SimData::PushConstant>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sim_datas_.push_constants_);

				uint32_t groupsSoftbodyStretch = CellDiv(count, 256);
				cmd.dispatch(groupsSoftbodyStretch, 1, 1);
			}
			TS(gp.timestamp_steps_);

			// solve_softbody_volume
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.softbody_volume)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_softbody_volume);

				sim_datas_.push_constants_.solve.compliance = sim_datas_.compliance.softbody_volume;
				cmd.pushConstants<SimData::PushConstant>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sim_datas_.push_constants_);

				uint32_t groupsSoftbodyVolume = CellDiv(sim_datas_.num_volumes, 256);
				cmd.dispatch(groupsSoftbodyVolume, 1, 1);
			}
			TS(gp.timestamp_steps_);

			// Solve Self Collision
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.self_collision && iter % sim_datas_.narrowphase_interval_ == 0)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_self_collision);
				sim_datas_.push_constants_.solve.compliance = sim_datas_.compliance.self_collision;
				cmd.pushConstants<SimData::PushConstant>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sim_datas_.push_constants_);
				cmd.dispatch(groupsTotal, 1, 1);
			}
			TS(gp.timestamp_steps_);

			// solve_inter_cloth_collision
			TS(gp.timestamp_steps_);
			if (sim_datas_.solver_config_.inter_collision)
			{
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_inter_cloth_collision);
				cmd.dispatch(groupsTotal, 1, 1);
			}
			TS(gp.timestamp_steps_);

			vku::ssboCompWtoCompRW(cmd, dSSBO.delta_x);
			vku::ssboCompWtoCompRW(cmd, dSSBO.delta_y);
			vku::ssboCompWtoCompRW(cmd, dSSBO.delta_z);
			vku::ssboCompWtoCompRW(cmd, dSSBO.delta_count);

			// Apply Deltas 
			TS(gp.timestamp_steps_);
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.apply_deltas);
			cmd.dispatch(groupsTotal, 1, 1);
			TS(gp.timestamp_steps_);
			vku::ssboCompWtoCompRW(cmd, pmSSBO.pred_position);
		}

		// Solve LRA
		TS(gp.timestamp_steps_);
		if (sim_datas_.solver_config_.lra && step == 0)
		{
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_lra);
			sim_datas_.push_constants_.solve.base = pm.clothes_[0].offset_particle;
			sim_datas_.push_constants_.solve.count = pm.num_cloth_particles_;
			cmd.pushConstants<SimData::PushConstant>(
				*pipeline_layouts_.common,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sim_datas_.push_constants_);

			uint32_t groupsCloth = CellDiv(sim_datas_.push_constants_.solve.count, 256);
			cmd.dispatch(groupsCloth, 1, 1);
		}
		TS(gp.timestamp_steps_);

		vku::ssboCompWtoCompRW(cmd, pmSSBO.pred_position);

		// Collide SDF
		TS(gp.timestamp_steps_);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.collide_sdf);
		cmd.dispatch(groupsTotal, 1, 1);
		TS(gp.timestamp_steps_);

		// Update Velocity
		TS(gp.timestamp_steps_);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.update_velocity);
		cmd.dispatch(groupsTotal, 1, 1);
		TS(gp.timestamp_steps_);
	}

	// Calculate Normals
	TS(gp.timestamp_steps_);
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.tri_normal);
	cmd.dispatch(groupsTri, 1, 1);
	TS(gp.timestamp_steps_);

	vku::Barrier2(cmd,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	TS(gp.timestamp_steps_);
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.vector_normal);
	cmd.dispatch(groupsTotal, 1, 1);
	TS(gp.timestamp_steps_);

	vku::ssboCompWtoVertR(cmd, pmSSBO.position);
	vku::ssboCompWtoVertR(cmd, pmSSBO.normal);

	TS(gp.timestamp_steps_); // End
	cmd.end();
}

void ClothXpbdPassGPU::CopySimDatas(const vk::raii::CommandBuffer& cmd)
{
	auto& pm = particle_manager_;

	vku::CopyStagingToSSBO(cmd, pm.ssbo_size_.position, pm.staging_mapped_.position, pm.positions_, pm.staging_.position, pm.ssbos_.position,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	vku::CopyStagingToSSBO(cmd, pm.ssbo_size_.pred_position, pm.staging_mapped_.pred_position, pm.pred_positions_, pm.staging_.pred_position, pm.ssbos_.pred_position,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	vku::CopyStagingToSSBO(cmd, pm.ssbo_size_.velocity, pm.staging_mapped_.velocity, pm.velocities_, pm.staging_.velocity, pm.ssbos_.velocity,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	vku::CopyStagingToSSBO(cmd, pm.ssbo_size_.inverse_mass, pm.staging_mapped_.inverse_mass, pm.inverse_masses_, pm.staging_.inverse_mass, pm.ssbos_.inverse_mass,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	vku::CopyStagingToSSBO(cmd, sim_datas_.ssbo_size_.edge, sim_datas_.staging_mapped_.edge, sim_datas_.edges, sim_datas_.staging_.edge, sim_datas_.ssbos_.edge,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	vku::CopyStagingToSSBO(cmd, sim_datas_.ssbo_size_.shear, sim_datas_.staging_mapped_.shear, sim_datas_.shears, sim_datas_.staging_.shear, sim_datas_.ssbos_.shear,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	vku::CopyStagingToSSBO(cmd, sim_datas_.ssbo_size_.bend, sim_datas_.staging_mapped_.bend, sim_datas_.bends, sim_datas_.staging_.bend, sim_datas_.ssbos_.bend,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	vku::CopyStagingToSSBO(cmd, sim_datas_.ssbo_size_.area, sim_datas_.staging_mapped_.area, sim_datas_.areas, sim_datas_.staging_.area, sim_datas_.ssbos_.area,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	vku::CopyStagingToSSBO(cmd, sim_datas_.ssbo_size_.volume, sim_datas_.staging_mapped_.volume, pm.volume_constraints, sim_datas_.staging_.volume, sim_datas_.ssbos_.volume,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
}

void ClothXpbdPassGPU::ResetTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene)
{
	auto& pm = particle_manager_;

	if (testScene.horizontal_drop)
	{
		testScene.horizontal_drop = false;
		ubo_.datas.sim_params.wind_enable = 0;

		float height = 5.0f;
		for (auto& cloth : pm.clothes_)
		{
			cloth.origin = glm::vec3(0.0f, height, 0.0f);
			cloth.angle_deg = 0.0f;
			cloth.axis = glm::vec3(1, 0, 0);
			pm.ResetCloth(cloth);
			cloth.angle_deg = 0.0f;

			height -= 1.0f;
		}
		sim_datas_.ResetConstraints(pm.positions_, pm.indices_);

		pm.ResetVolumeConstraint();

		CopySimDatas(cmd);
	}
	else if (testScene.vertical_drop)
	{
		testScene.vertical_drop = false;
		ubo_.datas.sim_params.wind_enable = 0;

		float height = 10.0f;
		for (auto& cloth : pm.clothes_)
		{
			cloth.origin = glm::vec3(0.0f, height, 0.0f);
			cloth.angle_deg = 90.0f;
			cloth.axis = glm::vec3(1, 0, 0);
			pm.ResetCloth(cloth);
			cloth.angle_deg = 0.0f;

			height -= (cloth.cloth_size.y + 1.0f);
		}
		sim_datas_.ResetConstraints(pm.positions_, pm.indices_);

		pm.ResetVolumeConstraint();

		CopySimDatas(cmd);
	}
	else if (testScene.pinned_corner)
	{
		testScene.pinned_corner = false;
		ubo_.datas.sim_params.wind_enable = 0;

		pm.clothes_[0].origin = glm::vec3(0.0f, 5.0f, 0.0f);
		pm.ResetCloth(pm.clothes_[0]);
		pm.clothes_[1].origin = glm::vec3(0.0f, 4.0f, 0.0f);
		pm.ResetCloth(pm.clothes_[1]);

		auto& cloth = pm.clothes_[2];
		cloth.origin = glm::vec3(0.0f, 3.0f, 0.0f);
		cloth.angle_deg = 0.0f;
		cloth.axis = glm::vec3(0, 1, 0);
		pm.ResetCloth(cloth);
		pm.inverse_masses_[cloth.offset_particle] = 0.0f;
		pm.inverse_masses_[cloth.offset_particle + cloth.nx1 - 1] = 0.0f;
		pm.inverse_masses_[cloth.offset_particle + (cloth.ny1 - 1) * cloth.nx1] = 0.0f;
		pm.inverse_masses_[cloth.offset_particle + (cloth.ny1 - 1) * cloth.nx1 + cloth.nx1 - 1] = 0.0f;
		sim_datas_.ResetConstraints(pm.positions_, pm.indices_);

		pm.ResetVolumeConstraint();

		CopySimDatas(cmd);
	}
	else if (testScene.top_pinned_corner)
	{
		testScene.top_pinned_corner = false;
		ubo_.datas.sim_params.wind_enable = 0;

		float height = 3.0f;
		for (auto& cloth : pm.clothes_)
		{
			cloth.origin = glm::vec3(0.0f, height, 0.0f);
			pm.ResetCloth(cloth);
			pm.inverse_masses_[cloth.offset_particle] = 0.0f;
			pm.inverse_masses_[cloth.offset_particle + cloth.nx1 - 1] = 0.0f;

			height -= 0.1f;
		}
		sim_datas_.ResetConstraints(pm.positions_, pm.indices_);

		pm.ResetVolumeConstraint();

		CopySimDatas(cmd);
	}
	else if (testScene.wind)
	{
		testScene.wind = false;
		ubo_.datas.sim_params.wind_enable = 1;

		float height = 8.0f;
		for (auto& cloth : pm.clothes_)
		{
			cloth.origin = glm::vec3(0.0f, height, 0.0f);
			//cloth.angle_deg = 90.0f;
			//cloth.axis = glm::vec3(1, 0, 0);
			pm.ResetCloth(cloth);
			cloth.angle_deg = 0.0f;
			pm.inverse_masses_[cloth.offset_particle] = 0.0f;
			pm.inverse_masses_[cloth.offset_particle + cloth.nx1 - 1] = 0.0f;

			height -= (cloth.cloth_size.y + 1.0f);
		}
		sim_datas_.ResetConstraints(pm.positions_, pm.indices_);

		pm.ResetVolumeConstraint();

		CopySimDatas(cmd);
	}
}

void ClothXpbdPassGPU::CopyColliders(const vk::raii::CommandBuffer& cmd)
{
	bool isCopyToGPU = false;

	uint32_t base = 0;
	for (uint32_t i = 0; i < model_manager_.models_.size(); i++)
	{
		auto& model = *model_manager_.models_[i];

		if (model.shape_collision_update_)
		{
			model.shape_collision_update_ = false;

			isCopyToGPU = true;

			model.UpdateShapeColliders();

			//std::cout <<" Update " << model.name_ << std::endl;

			for (uint32_t j = 0; j < model.shape_colliders_.size(); j++)
				sim_datas_.colliders[base + j] = model.shape_colliders_[j];

		}
		base += model.shape_colliders_.size();

		if (model.capsule_collision_update_)
		{
			model.capsule_collision_update_ = false;

			isCopyToGPU = true;

			model.UpdateCapsuleCollidersFromBones();

			for (uint32_t j = 0; j < model.capsule_colliders_.size(); j++)
				sim_datas_.colliders[base + j] = model.capsule_colliders_[j];
		}
		base += model.capsule_colliders_.size();
	}

	if (isCopyToGPU)
	{
		vku::CopyStagingToSSBO(cmd, sim_datas_.ssbo_size_.collider, sim_datas_.staging_mapped_.collider, sim_datas_.colliders, sim_datas_.staging_.collider, sim_datas_.ssbos_.collider,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
	}

}

void ClothXpbdPassGPU::CreateCommandBuffers()
{
	cmds_.clear();
	vk::CommandBufferAllocateInfo allocInfo{};
	allocInfo.commandPool = *context_.command_pool_;
	allocInfo.level = vk::CommandBufferLevel::ePrimary;
	allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
	cmds_ = vk::raii::CommandBuffers(context_.device_, allocInfo);
}

void ClothXpbdPassGPU::CreateDescriptorSetLayout()
{
	// Sim params UBO
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eCompute, nullptr)
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.sim_params = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Cloth compute
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 8, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 9, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 10, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 11, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 12, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 13, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 14, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 15, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 16, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 17, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 18, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 19, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 20, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 21, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 22, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 23, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 24, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 25, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 26, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 27, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 28, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
			vk::DescriptorSetLayoutBinding{ 29, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
		};
		counts_.sb += 30;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.cloth_compute = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

}

void ClothXpbdPassGPU::CreateDescriptorPools()
{
	std::vector<vk::DescriptorPoolSize> poolSizes;

	if (counts_.ubo > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer, counts_.ubo);
	}
	if (counts_.ubo_dynamic > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eUniformBufferDynamic, counts_.ubo_dynamic);
	}
	if (counts_.sampler > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler, counts_.sampler);
	}
	if (counts_.sb > 0) {
		poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer, counts_.sb);
	}

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = counts_.layout,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};

	descriptor_pool_ = vk::raii::DescriptorPool(context_.device_, poolInfo);
}

void ClothXpbdPassGPU::CreateUniformBuffers()
{
	// Sim params UBO
	{
		ubo_.ubos.sim_params.clear();
		ubo_.memories.sim_params.clear();
		ubo_.mapped.sim_params = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_.size.sim_params = (sizeof(SimUBO::Data::SimParams) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_.size.sim_params * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubo_.ubos.sim_params = std::move(buffer);
		ubo_.memories.sim_params = std::move(bufferMem);
		ubo_.mapped.sim_params = ubo_.memories.sim_params.mapMemory(0, totalSize);
	}

}

void ClothXpbdPassGPU::CreateClothConstraintDatas()
{
	auto& pm = particle_manager_;
	auto& d = sim_datas_;

	sim_datas_.cloth_particles_ = pm.num_cloth_particles_;
	sim_datas_.cloth_indices_ = pm.num_cloth_indices_;
	uint32_t N = sim_datas_.cloth_particles_;

	sim_datas_.BuildStretchConstraints(pm.positions_, pm.indices_, pm.clothes_);
	sim_datas_.BuildShearConstraints(pm.positions_, pm.indices_, pm.clothes_);
	sim_datas_.BuildBendConstraints(pm.positions_, pm.indices_, pm.clothes_);
	sim_datas_.BuildAreaConstraints(pm.positions_, pm.indices_, pm.clothes_);
	sim_datas_.BuildLRAConstraints(pm.positions_, pm.indices_, pm);
}

void ClothXpbdPassGPU::CreateSoftBodyConstraintDatas()
{
	auto& pm = particle_manager_;

	sim_datas_.softbody_particles_ = pm.num_softbody_particles_;
	sim_datas_.softbody_indices_ = pm.num_softbody_indices_;

	for (uint32_t i = 0; i < pm.softbodies_.size(); i++)
	{
		auto& tetmesh = pm.softbodies_[i].tetmesh;

		uint32_t offsetParticle = particle_manager_.softbodies_[i].offset_particle;
		std::unordered_set<SimData::EdgeKey, SimData::EdgeKeyHash> edgeSet;

		auto pushEdge = [&](uint32_t i, uint32_t j) {
			uint32_t a = std::min(i, j);
			uint32_t b = std::max(i, j);
			SimData::EdgeKey e{ a, b };
			if (edgeSet.insert(e).second) {
				SimData::Edge c;
				c.i = a + offsetParticle;
				c.j = b + offsetParticle;
				c.rest = length(pm.positions_[c.i] - pm.positions_[c.j]);
				sim_datas_.edges.push_back(c);
			}
			};

		for (auto& tet : tetmesh.tets) {
			uint32_t i0 = tet.x;
			uint32_t i1 = tet.y;
			uint32_t i2 = tet.z;
			uint32_t i3 = tet.w;

			pushEdge(i0, i1);
			pushEdge(i0, i2);
			pushEdge(i0, i3);
			pushEdge(i1, i2);
			pushEdge(i1, i3);
			pushEdge(i2, i3);
		}
	}
	sim_datas_.num_edges = static_cast<uint32_t>(sim_datas_.edges.size());
	sim_datas_.num_softbody_edges = sim_datas_.num_edges - sim_datas_.num_cloth_edges;

	sim_datas_.pass_offsets.push_back(sim_datas_.num_edges);

	sim_datas_.num_volumes = pm.volume_constraints.size();
}

void ClothXpbdPassGPU::CreateColiiders()
{
	for (auto& model : model_manager_.models_)
	{
		for (auto c : model->shape_colliders_)
			sim_datas_.colliders.push_back(c);

		for (auto c : model->capsule_colliders_)
			sim_datas_.colliders.push_back(c);
	}

	sim_datas_.num_colliders = sim_datas_.colliders.size();
}

void ClothXpbdPassGPU::CreateSSBOBuffers()
{
	// Self Collision
	auto& pm = particle_manager_;

	uint32_t N = sim_datas_.total_particles_;
	uint32_t tableSize = N;
	uint32_t maxNeighbors = 16;
	ubo_.datas.sim_params.num_tables = tableSize;
	ubo_.datas.sim_params.cell_size = pm.default_cloth_spacing_;
	ubo_.datas.sim_params.collision_radius = ubo_.datas.sim_params.cell_size;
	ubo_.datas.sim_params.max_neighbors = maxNeighbors;

	std::vector<float > deltaX(N, 0.0f), deltaY(N, 0.0f), deltaZ(N, 0.0f);
	std::vector<uint32_t>  delta_count(N, 0);

	// delta X
	sim_datas_.ssbo_size_.delta_x = sizeof(float) * N;
	vku::CreateSSBO("Delta X", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.delta_x,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		deltaX,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.delta_x, sim_datas_.ssbo_memories_.delta_x);

	// delta Y
	sim_datas_.ssbo_size_.delta_y = sizeof(float) * N;
	vku::CreateSSBO("Delta Y", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.delta_y,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		deltaY,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.delta_y, sim_datas_.ssbo_memories_.delta_y);

	// delta Z
	sim_datas_.ssbo_size_.delta_z = sizeof(float) * N;
	vku::CreateSSBO("Delta Z", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.delta_z,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		deltaZ,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.delta_z, sim_datas_.ssbo_memories_.delta_z);

	// delta count
	sim_datas_.ssbo_size_.delta_count = sizeof(uint32_t) * N;
	vku::CreateSSBO("Delta Count", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.delta_count,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		delta_count,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.delta_count, sim_datas_.ssbo_memories_.delta_count);

	// edges
	sim_datas_.ssbo_size_.edge = sizeof(SimData::Edge) * sim_datas_.num_edges;
	vku::CreateSSBO("Edge", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.edge,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		sim_datas_.edges,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.edge, sim_datas_.ssbo_memories_.edge,
		&sim_datas_.staging_.edge, &sim_datas_.staging_memories_.edge);
	sim_datas_.staging_mapped_.edge = sim_datas_.staging_memories_.edge.mapMemory(0, sim_datas_.ssbo_size_.edge);

	// Bend
	sim_datas_.ssbo_size_.bend = sizeof(SimData::Bend) * sim_datas_.num_bends;
	vku::CreateSSBO("Bend", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.bend,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		sim_datas_.bends,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.bend, sim_datas_.ssbo_memories_.bend,
		&sim_datas_.staging_.bend, &sim_datas_.staging_memories_.bend);
	sim_datas_.staging_mapped_.bend = sim_datas_.staging_memories_.bend.mapMemory(0, sim_datas_.ssbo_size_.bend);

	// Shear
	sim_datas_.ssbo_size_.shear = sizeof(SimData::Shear) * sim_datas_.num_shears;
	vku::CreateSSBO("Shear", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.shear,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		sim_datas_.shears,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.shear, sim_datas_.ssbo_memories_.shear,
		&sim_datas_.staging_.shear, &sim_datas_.staging_memories_.shear);
	sim_datas_.staging_mapped_.shear = sim_datas_.staging_memories_.shear.mapMemory(0, sim_datas_.ssbo_size_.shear);

	// grab_counter
	struct GrabState {
		uint32_t id = 0;
		uint32_t t_bits = 0;
		float	 t = 0.0f;
	};
	std::vector<GrabState> grabState(1);
	sim_datas_.ssbo_size_.grab_state = sizeof(GrabState) * grabState.size();
	vku::CreateSSBO("Grab Counter", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.grab_state,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		grabState,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.grab_state, sim_datas_.ssbo_memories_.grab_state);

	// Area
	sim_datas_.ssbo_size_.area = sizeof(SimData::Area) * sim_datas_.num_areas;
	vku::CreateSSBO("Area", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.area,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		sim_datas_.areas,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.area, sim_datas_.ssbo_memories_.area,
		&sim_datas_.staging_.area, &sim_datas_.staging_memories_.area);
	sim_datas_.staging_mapped_.area = sim_datas_.staging_memories_.area.mapMemory(0, sim_datas_.ssbo_size_.area);

	// volume
	sim_datas_.ssbo_size_.volume = sizeof(ParticleManager::Volume) * sim_datas_.num_volumes;
	vku::CreateSSBO("Volume", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.volume,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		particle_manager_.volume_constraints,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.volume, sim_datas_.ssbo_memories_.volume,
		&sim_datas_.staging_.volume, &sim_datas_.staging_memories_.volume);
	sim_datas_.staging_mapped_.volume = sim_datas_.staging_memories_.volume.mapMemory(0, sim_datas_.ssbo_size_.volume);

	// collider
	sim_datas_.ssbo_size_.collider = sizeof(Collider) * sim_datas_.num_colliders;
	vku::CreateSSBO("Collider", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.collider,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		sim_datas_.colliders,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.collider, sim_datas_.ssbo_memories_.collider,
		&sim_datas_.staging_.collider, &sim_datas_.staging_memories_.collider);
	sim_datas_.staging_mapped_.collider = sim_datas_.staging_memories_.collider.mapMemory(0, sim_datas_.ssbo_size_.collider);

	// delta v
	sim_datas_.ssbo_size_.delta_v = sizeof(glm::vec4) * N;
	std::vector<glm::vec4> deltaV(N, glm::vec4(0));
	vku::CreateSSBO("Delta V", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.delta_v,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		deltaV,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.delta_v, sim_datas_.ssbo_memories_.delta_v);

	// lra_ids
	sim_datas_.ssbo_size_.lra_ids = sizeof(uint32_t) * sim_datas_.lra_ids.size();
	vku::CreateSSBO("lra_ids", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.lra_ids,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		sim_datas_.lra_ids,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.lra_ids, sim_datas_.ssbo_memories_.lra_ids);

	// lra_r
	sim_datas_.ssbo_size_.lra_r = sizeof(float) * sim_datas_.lra_r.size();
	vku::CreateSSBO("lra_r", context_.physical_device_, context_.device_, context_.queue_, context_.command_pool_,
		sim_datas_.ssbo_size_.lra_r,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		sim_datas_.lra_r,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		sim_datas_.ssbos_.lra_r, sim_datas_.ssbo_memories_.lra_r);
}

void ClothXpbdPassGPU::CreateDescriptorSets()
{
	// Sim Params
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.sim_params
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.sim_params = std::move(sets.front());

		vk::DescriptorBufferInfo simParamsUboInfo{ *ubo_.ubos.sim_params, 0, sizeof(SimUBO::Data::SimParams) };
		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.sim_params,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &simParamsUboInfo
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}


	// Cloth Compute
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.cloth_compute
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.cloth_compute = std::move(sets.front());

		auto& pmSSBO = particle_manager_.ssbos_;
		auto& dSSBO = sim_datas_.ssbos_;

		vk::DescriptorBufferInfo positions(*pmSSBO.position, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo predPositions(*pmSSBO.pred_position, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo velocities(*pmSSBO.velocity, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo inverseMass(*pmSSBO.inverse_mass, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaX(*dSSBO.delta_x, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaY(*dSSBO.delta_y, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaZ(*dSSBO.delta_z, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaCount(*dSSBO.delta_count, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo edge(*dSSBO.edge, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo shear(*dSSBO.shear, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo bend(*dSSBO.bend, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo grabState(*dSSBO.grab_state, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo area(*dSSBO.area, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo hash(*pmSSBO.particle_hash, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo sortedIndice(*pmSSBO.sorted_indice, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo start(*pmSSBO.start, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo end(*pmSSBO.end, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo neighbor(*pmSSBO.neighbor, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo neighborLambda(*pmSSBO.neighbor_lambda, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo index(*pmSSBO.index, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo normal(*pmSSBO.normal, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo triNormal(*pmSSBO.tri_normals, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo vertexTriOffset(*pmSSBO.vertex_tri_offsets, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo vertexTriIndice(*pmSSBO.vertex_tri_indices, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo volume(*dSSBO.volume, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo collider(*dSSBO.collider, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo collisionMask(*pmSSBO.collision_masks_, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaV(*dSSBO.delta_v, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo lraIds(*dSSBO.lra_ids, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo lraR(*dSSBO.lra_r, 0, VK_WHOLE_SIZE);

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &positions
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &predPositions
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &velocities
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &inverseMass
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 4,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &deltaX
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 5,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &deltaY
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 6,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &deltaZ
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 7,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &deltaCount
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 8,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &edge
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 9,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &shear
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 10,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &bend
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 11,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &grabState
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 12,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &area
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 13,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &hash
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 14,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &sortedIndice
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 15,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &start
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 16,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &end
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 17,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &neighbor
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 18,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &neighborLambda
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 19,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &index
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 20,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &normal
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 21,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &triNormal
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 22,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &vertexTriOffset
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 23,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &vertexTriIndice
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 24,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &volume
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 25,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &collider
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 26,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &collisionMask
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 27,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &deltaV
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 28,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &lraIds
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_compute,
				.dstBinding = 29,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &lraR
			},
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}
}

void ClothXpbdPassGPU::CreateComputePipelineLayouts()
{
	// common pipeline layout
	{
		std::array<vk::DescriptorSetLayout, 2> setLayouts(
			*set_layouts_.sim_params,
			*set_layouts_.cloth_compute
		);

		vk::PushConstantRange pcRange{
			.stageFlags = vk::ShaderStageFlagBits::eCompute,
			.offset = 0,
			.size = static_cast<uint32_t>(sizeof(SimData::PushConstant))
		};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = 2,
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcRange,
		};
		pipeline_layouts_.common = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);
	}
}

void ClothXpbdPassGPU::CreateComputePipelines()
{
	// wind
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/wind.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.wind = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// clear_lambdas
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/clear_lambdas.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.clear_lambdas = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// integrate
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/integrate.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.integrate = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// solve_stretch
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/solve_stretch.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_stretch = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// solve_shear
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/solve_shear.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.solve_shear = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// solve_bend
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/solve_bend.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_bend = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// solve_area
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/solve_area.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_area = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// solve_softbody_stretch
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/solve_softbody_stretch.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_softbody_stretch = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// solve_softbody_volume
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/solve_softbody_volume.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_softbody_volume = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// apply deltas
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/apply_deltas.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.apply_deltas = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// collide_sdf
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/collide_sdf.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.collide_sdf = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// update velocity
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/update_velocity.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.update_velocity = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// build_hash
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/build_hash.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.build_hash = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);

	}

	// build_cell
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/build_cell.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.build_cell = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);

	}

	// build_neighbor
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/build_neighbor.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.build_neighbor = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// solve_self_collision
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/solve_self_collision.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.solve_self_collision = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// tri_normal
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/tri_normal.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.tri_normal = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// vertex_normal
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/vertex_normal.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.vector_normal = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// solve_inter_cloth_collision
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/solve_inter_cloth_collision.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.solve_inter_cloth_collision = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}

	// solve_inter_cloth_collision
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context_.device_, vku::ReadFile("shaders/spv/solve_lra.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.solve_lra = vk::raii::Pipeline(context_.device_, nullptr, pipelineInfo);
	}
}

void ClothXpbdPassGPU::CreateVrdxSorter()
{
	VrdxSorterCreateInfo info{};
	info.physicalDevice = *context_.physical_device_;
	info.device = *context_.device_;
	info.pipelineCache = VK_NULL_HANDLE;

	vrdxCreateSorter(&info, &radix_.sorter);

	VrdxSorterStorageRequirements req{};
	vrdxGetSorterKeyValueStorageRequirements(radix_.sorter, sim_datas_.total_particles_, &req);
	radix_.storage_size = req.size;

	vku::CreateBuffer(context_.physical_device_,
		context_.device_,
		req.size,
		vk::BufferUsageFlags(req.usage),
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		radix_.storage_buffer,
		radix_.storage_memory);
}

void ClothXpbdPassGPU::CalculateGpuTime()
{
	gpu_profiler_->CalculateGpuTime(sim_datas_);
}