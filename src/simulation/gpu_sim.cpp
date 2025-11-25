#include "context.h"
#include "swapchain.h"
#include "texture.h"
#include "texture_manager.h"
#include "model.h"
#include "model_manager.h"
#include "vulkan_utils.h"
#include "ray.h"
#include "mouse_interactor.h"

#include "gpu_sim.h"

GpuSim::GpuSim(
	Context& context,
	Swapchain& swapchain,
	TextureManager& textureManager,
	ModelManager& modelManager,
	vk::raii::DescriptorSetLayout& globalSetLayout,
	std::vector<vk::Format>& formats,
	vk::raii::DescriptorSetLayout& tex2DSetLayout)
{
	CreateDescriptorSetLayout(context);
	CreateDescriptorPools(context);
	CreateUniformBuffers(context, modelManager);
	CreateSSBOBuffers(context);
	CreateDescriptorSets(context, textureManager);
	CreateComputePipelines(context);
	CreateGraphicsPipelines(context, globalSetLayout, formats, tex2DSetLayout);
}

void GpuSim::UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize)
{
	if (mouseInteractor.is_left_button_down_event)
	{
		push_constants_.mouse_interact.select_mode = 1;

		Ray ray = mouseInteractor.CalculateMouseRay(camera, viewportSize);

		push_constants_.mouse_interact.ray_origin = ray.origin;
		push_constants_.mouse_interact.ray_dir = ray.direction;

	}
	else if (!mouseInteractor.is_left_button_down_event && mouseInteractor.is_left_down)
	{
		push_constants_.mouse_interact.select_mode = 2;

		Ray ray = mouseInteractor.CalculateMouseRay(camera, viewportSize);

		push_constants_.mouse_interact.ray_origin = ray.origin;
		push_constants_.mouse_interact.ray_dir = ray.direction;
	}
	else
		push_constants_.mouse_interact.select_mode = 0;
}

void GpuSim::UpdateComputeUBO(uint32_t currentFrame, std::unique_ptr<Model>& model)
{
	ubo_.datas_.sim_params.dt = 1.0f / datas_.frame_dt_ / datas_.substeps_;
	ubo_.datas_.sim_params.num_particles = datas_.particles_size_;
	ubo_.datas_.sim_params.num_edges = datas_.edge_size_;
	ubo_.datas_.sim_params.num_bends = datas_.bend_size_;
	ubo_.datas_.sim_params.num_shears = datas_.shear_size_;
	ubo_.datas_.sim_params.num_areas = datas_.area_size_;
	ubo_.datas_.sim_params.sphere_center = glm::vec4(model->position_, 0.0f);
	ubo_.datas_.sim_params.sphere_radius = model->radius_;

	const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * ubo_.size_.sim_params);
	auto* dst = static_cast<std::byte*>(ubo_.mapped_.sim_params) + baseOffset;

	std::memcpy(dst, &ubo_.datas_.sim_params, sizeof(SimUBO::Data::SimParams));
}

void GpuSim::RecordCompute(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::QueryPool& timestampPool, uint32_t& timestampSteps, vku::TestScene& testScene)
{
	cmd.reset();
	cmd.begin({});

	UpdateTestScene(cmd, testScene);

	timestampSteps = 0;
	uint32_t kSlotsPerIterPair = datas_.substeps_ * (4 + iteration_timestamp_count_ * datas_.iterations_ + 2);
	const auto stage = vk::PipelineStageFlagBits2::eComputeShader;
	auto TS = [&](uint32_t& idx) {
		cmd.writeTimestamp2(stage, *timestampPool, idx++);
		};

	cmd.resetQueryPool(*timestampPool, 0, kSlotsPerIterPair);

	uint32_t simparamOffset = currentFrame * static_cast<uint32_t>(ubo_.size_.sim_params);
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eCompute, pipeline_layouts_.common,
		0,
		{ sets_.sim_params, sets_.cloth_compute },
		{ simparamOffset }
	);

	auto ceil_div = [](uint32_t n, uint32_t d) { return (n + d - 1) / d; };
	uint32_t groupsP = ceil_div(datas_.particles_size_, 256u);
	uint32_t groupsEdges = ceil_div(datas_.edge_size_, 256u);

	for (uint32_t step = 0; step < datas_.substeps_; step++)
	{
		// Integrate
		TS(timestampSteps);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.integrate);
		cmd.pushConstants<PushConstant::MouseInteract>(
			*pipeline_layouts_.common,
			vk::ShaderStageFlagBits::eCompute,
			static_cast<uint32_t>(sizeof(PushConstant::Solve)), // offset
			push_constants_.mouse_interact);
		cmd.dispatch(groupsP, 1, 1);
		TS(timestampSteps);
		vku::ssboCompWtoCompRW(cmd, ssbos_.pred_position);

		// Clear Lambdas
		TS(timestampSteps);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.clear_lambdas);
		cmd.dispatch(groupsEdges, 1, 1);
		TS(timestampSteps);

		for (uint32_t iter = 0; iter < datas_.iterations_; iter++)
		{
			// Solve Coloring - Stretch
			TS(timestampSteps);
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_stretch);

			for (uint32_t c = 0; c < 4; ++c) {
				uint32_t base = datas_.pass_offsets[c];
				uint32_t count = datas_.pass_offsets[c + 1] - datas_.pass_offsets[c];
				if (!count) continue;

				push_constants_.solve.base = base;
				push_constants_.solve.count = count;
				push_constants_.solve.compliance = datas_.compliance_.stretch;
				push_constants_.solve.beta = datas_.beta_.stretch;

				cmd.pushConstants<PushConstant::Solve>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute, 0u, push_constants_.solve);

				uint32_t groups = (count + 256 - 1) / 256;
				cmd.dispatch(groups, 1, 1);
				vku::ssboCompWtoCompRW(cmd, ssbos_.pred_position);
			}
			TS(timestampSteps);

			{
				// Solve Shear
				TS(timestampSteps);
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_shear);
				push_constants_.solve.compliance = datas_.compliance_.shear;
				push_constants_.solve.beta = datas_.beta_.shear;

				cmd.pushConstants<PushConstant::Solve>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute, 0u, push_constants_.solve);

				uint32_t group = (datas_.shear_size_ + 256 - 1) / 256;
				cmd.dispatch(group, 1, 1);
				TS(timestampSteps);
			}

			{
				// Solve Bend
				TS(timestampSteps);
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_bend);
				uint32_t base = 0;
				uint32_t count = datas_.bend_size_;
				push_constants_.solve.base = base;
				push_constants_.solve.count = count;
				push_constants_.solve.compliance = datas_.compliance_.bend;
				push_constants_.solve.beta = datas_.beta_.bend;
				cmd.pushConstants<PushConstant::Solve>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute, 0u, push_constants_.solve);

				uint32_t group = (count + 256 - 1) / 256;
				cmd.dispatch(group, 1, 1);
				TS(timestampSteps);
			}

			{
				// Solve Area
				TS(timestampSteps);
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_area);
				uint32_t base = 0;
				uint32_t count = datas_.area_size_;
				push_constants_.solve.base = base;
				push_constants_.solve.count = count;
				push_constants_.solve.compliance = datas_.compliance_.area;
				push_constants_.solve.beta = datas_.beta_.area;
				cmd.pushConstants<PushConstant::Solve>(
					*pipeline_layouts_.common,
					vk::ShaderStageFlagBits::eCompute, 0u, push_constants_.solve);

				uint32_t group = (count + 256 - 1) / 256;
				cmd.dispatch(group, 1, 1);
				TS(timestampSteps);
			}

			// Collide SDF
			TS(timestampSteps);
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.collide_sdf);
			cmd.dispatch(groupsP, 1, 1);
			TS(timestampSteps);

			vku::ssboCompWtoCompRW(cmd, ssbos_.delta_x);
			vku::ssboCompWtoCompRW(cmd, ssbos_.delta_y);
			vku::ssboCompWtoCompRW(cmd, ssbos_.delta_z);
			vku::ssboCompWtoCompRW(cmd, ssbos_.delta_count);

			// Apply Deltas 
			TS(timestampSteps);
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.apply_deltas);
			cmd.dispatch(groupsP, 1, 1);
			TS(timestampSteps);
			vku::ssboCompWtoCompRW(cmd, ssbos_.pred_position);
		}

		// Update Velocity
		TS(timestampSteps);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.update_velocity);
		cmd.dispatch(groupsP, 1, 1);
		TS(timestampSteps);
		vku::ssboCompWtoVertR(cmd, ssbos_.position);
	}
	cmd.end();
}

void GpuSim::UpdateGraphicsUBO(uint32_t currentFrame)
{
	const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * ubo_.size_.render);
	auto* dst = static_cast<std::byte*>(ubo_.mapped_.render) + baseOffset;

	std::memcpy(dst, &ubo_.datas_.render, sizeof(SimUBO::Data::Render));
}

void GpuSim::RecordGraphics(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset, vku::PolygonMode mode,
	vk::raii::DescriptorSet& tex2DSet)
{
	// Cloth
	{
		if (mode == vku::PolygonMode::WIREFRAME)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.cloth_wireframe);
		else if (mode == vku::PolygonMode::POINT)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.cloth_point);
		else
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.cloth_solid);

		// Global Set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.cloth_graphics,
			0,
			{ *globalSet },
			{ globalOffset }
		);

		// Graphics
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.cloth_graphics,
			1,
			{ *sets_.cloth_graphics },
			{ }
		);

		// Render
		const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * ubo_.size_.render);
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.cloth_graphics,
			2,
			{ *sets_.render },
			{ baseOffset }
		);

		// Tex
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.cloth_graphics,
			3,
			{ *tex2DSet },
			{ }
		);

		cmd.pushConstants<PushConstant::ClothRender>(
			*pipeline_layouts_.cloth_graphics,
			vk::ShaderStageFlagBits::eVertex,
			/*offset=*/0,
			push_constants_.cloth_render
		);

		cmd.bindIndexBuffer(*index_buffer_, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(datas_.indices_size_, 1, 0, 0, 0);
	}
}

void GpuSim::CopyDatas(const vk::raii::CommandBuffer& cmd)
{
	vku::CopyStagingToSSBO(cmd, ssbo_size_.position, staging_mapped_.position, datas_.positions, staging_.position, ssbos_.position,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.pred_position, staging_mapped_.pred_position, datas_.pred_positions, staging_.pred_position, ssbos_.pred_position,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.velocity, staging_mapped_.velocity, datas_.velocities, staging_.velocity, ssbos_.velocity,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.inverse_mass, staging_mapped_.inverse_mass, datas_.inverse_masses, staging_.inverse_mass, ssbos_.inverse_mass,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.edge, staging_mapped_.edge, datas_.edges, staging_.edge, ssbos_.edge,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.shear, staging_mapped_.shear, datas_.shears, staging_.shear, ssbos_.shear,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.bend, staging_mapped_.bend, datas_.bends, staging_.bend, ssbos_.bend,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);
}

void GpuSim::UpdateTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene)
{
	datas_.spacing_x_ = datas_.cloth_size_.x / datas_.nx_;
	datas_.spacing_y_ = datas_.cloth_size_.y / datas_.ny_;

	if (testScene.sphereCollision)
	{
		testScene.sphereCollision = false;

		const int nxCells = datas_.nx_;
		const int nyCells = datas_.ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * datas_.nx_) * datas_.spacing_x_;
				float py = datas_.cloth_height_;
				float pz = (y - 0.5f * datas_.ny_) * datas_.spacing_y_;
				datas_.positions[id] = { px, py, pz, 0.0f };
				datas_.velocities[id] = glm::vec4(0.0f);
				datas_.pred_positions[id] = datas_.positions[id];
			}
		}

		datas_.ResetConstraints();
		CopyDatas(cmd);
	}
	else if (testScene.pinnedCorner)
	{
		testScene.pinnedCorner = false;

		const int nxCells = datas_.nx_;
		const int nyCells = datas_.ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * datas_.nx_) * datas_.spacing_x_;
				float py = datas_.cloth_height_;
				float pz = (y - 0.5f * datas_.ny_) * datas_.spacing_y_;
				datas_.positions[id] = { px, py, pz, 0.0f };
				datas_.velocities[id] = glm::vec4(0.0f);
				datas_.pred_positions[id] = datas_.positions[id];
			}
		}

		datas_.ResetConstraints();

		datas_.inverse_masses[0] = 0.0f;
		datas_.inverse_masses[nx1 - 1] = 0.0f;
		datas_.inverse_masses[nx1 * (ny1 - 1)] = 0.0f;
		datas_.inverse_masses[nx1 * (ny1 - 1) + nx1 - 1] = 0.0f;

		CopyDatas(cmd);
	}
	else if (testScene.topPinnedCorner)
	{
		testScene.topPinnedCorner = false;

		const int nxCells = datas_.nx_;
		const int nyCells = datas_.ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * datas_.nx_) * datas_.spacing_x_;
				float py = datas_.cloth_height_;
				float pz = (y - 0.5f * datas_.ny_) * datas_.spacing_y_;
				datas_.positions[id] = { px, py, pz, 0.0f };
				datas_.velocities[id] = glm::vec4(0.0f);
				datas_.pred_positions[id] = datas_.positions[id];
			}
		}

		datas_.ResetConstraints();

		datas_.inverse_masses[0] = 0.0f;
		datas_.inverse_masses[nx1 - 1] = 0.0f;

		CopyDatas(cmd);
	}
}

void GpuSim::CreateDescriptorSetLayout(Context& context)
{
	// Sim params UBO
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eCompute, nullptr)
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.sim_params = vk::raii::DescriptorSetLayout(context.device_, layoutInfo);
	}

	// Render
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr)
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.render = vk::raii::DescriptorSetLayout(context.device_, layoutInfo);
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
		};
		counts_.sb += 13;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.cloth_compute = vk::raii::DescriptorSetLayout(context.device_, layoutInfo);
	}

	// Cloth graphics
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex }
		};
		counts_.sb += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.cloth_graphics = vk::raii::DescriptorSetLayout(context.device_, layoutInfo);
	}
}

void GpuSim::CreateDescriptorPools(Context& context)
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

	descriptor_pool_ = vk::raii::DescriptorPool(context.device_, poolInfo);
}

void GpuSim::CreateUniformBuffers(Context& context,
	ModelManager& modelManager)
{
	// Sim params UBO
	{
		ubo_.ubos_.sim_params.clear();
		ubo_.memories_.sim_params.clear();
		ubo_.mapped_.sim_params = nullptr;

		auto limits = context.physical_device_.getProperties().limits;
		ubo_.size_.sim_params = (sizeof(SimUBO::Data::SimParams) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_.size_.sim_params * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context.physical_device_, context.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubo_.ubos_.sim_params = std::move(buffer);
		ubo_.memories_.sim_params = std::move(bufferMem);
		ubo_.mapped_.sim_params = ubo_.memories_.sim_params.mapMemory(0, totalSize);
	}

	// Render UBO
	{
		ubo_.ubos_.render.clear();
		ubo_.memories_.render.clear();
		ubo_.mapped_.render = nullptr;

		auto limits = context.physical_device_.getProperties().limits;
		ubo_.size_.render = (sizeof(SimUBO::Data::Render) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_.size_.render * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context.physical_device_, context.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubo_.ubos_.render = std::move(buffer);
		ubo_.memories_.render = std::move(bufferMem);
		ubo_.mapped_.render = ubo_.memories_.render.mapMemory(0, totalSize);

		ubo_.datas_.render.albedo_enable = 1;
		ubo_.datas_.render.albedo_idx = 0;
	}
}

void GpuSim::CreateSSBOBuffers(Context& context)
{
	const int nxCells = datas_.nx_;
	const int nyCells = datas_.ny_;
	const int nx1 = nxCells + 1;
	const int ny1 = nyCells + 1;

	auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

	const uint32_t N = nx1 * ny1;
	datas_.particles_size_ = N;

	datas_.spacing_x_ = datas_.cloth_size_.x / datas_.nx_;
	datas_.spacing_y_ = datas_.cloth_size_.y / datas_.ny_;

	datas_.positions.resize(N);
	datas_.velocities.resize(N);
	datas_.inverse_masses.resize(N);
	datas_.pred_positions.resize(N);
	std::vector<float > deltaX(N, 0.0f), deltaY(N, 0.0f), deltaZ(N, 0.0f);
	std::vector<uint32_t>  delta_count(N, 0);

	// Set positions, velocities, pred_positions
	for (int y = 0; y < ny1; ++y) {
		for (int x = 0; x < nx1; ++x) {
			uint32_t id = vid(x, y);
			float px = (-0.5f * datas_.nx_ + x) * datas_.spacing_x_;
			float py = datas_.cloth_height_;
			float pz = (-0.5f * datas_.ny_ + y) * datas_.spacing_y_;

			datas_.positions[id] = { px, py, pz, 0.0f };
			datas_.velocities[id] = glm::vec4(0);
			datas_.pred_positions[id] = datas_.positions[id];
		}
	}

	// Set indices
	datas_.indices.reserve(datas_.nx_ * datas_.ny_ * 6);
	for (int y = 0; y < datas_.ny_; ++y) {
		for (int x = 0; x < datas_.nx_; ++x) {
			uint32_t i0 = vid(x, y);
			uint32_t i1 = vid(x + 1, y);
			uint32_t i2 = vid(x, y + 1);
			uint32_t i3 = vid(x + 1, y + 1);
			datas_.indices.push_back(i0); datas_.indices.push_back(i2); datas_.indices.push_back(i1);
			datas_.indices.push_back(i1); datas_.indices.push_back(i2); datas_.indices.push_back(i3);
		}
	}
	datas_.indices_size_ = static_cast<uint32_t>(datas_.indices.size());
	vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, datas_.indices, index_buffer_, index_buffer_memory_);

	datas_.masses.resize(N, 0.0f);

	// Total area
	float totalArea = 0.0f;
	for (size_t t = 0; t < datas_.indices_size_; t += 3) {
		uint32_t i0 = datas_.indices[t + 0];
		uint32_t i1 = datas_.indices[t + 1];
		uint32_t i2 = datas_.indices[t + 2];

		glm::vec3 p0 = glm::vec3(datas_.positions[i0]);
		glm::vec3 p1 = glm::vec3(datas_.positions[i1]);
		glm::vec3 p2 = glm::vec3(datas_.positions[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2)); // Triangle area
		totalArea += area;
	}

	float totalMassTarget = datas_.mass_;

	// Area zero defence
	float density = 0.0f;
	if (totalArea > 0.0f) {
		density = totalMassTarget / totalArea; // kg/m©÷
	}

	// Distribute mass to each triangle in proportion to area
	for (size_t t = 0; t < datas_.indices_size_; t += 3) {
		uint32_t i0 = datas_.indices[t + 0];
		uint32_t i1 = datas_.indices[t + 1];
		uint32_t i2 = datas_.indices[t + 2];

		glm::vec3 p0 = glm::vec3(datas_.positions[i0]);
		glm::vec3 p1 = glm::vec3(datas_.positions[i1]);
		glm::vec3 p2 = glm::vec3(datas_.positions[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2));

		float triMass = density * area;

		float share = triMass / 3.0f;
		datas_.masses[i0] += share;
		datas_.masses[i1] += share;
		datas_.masses[i2] += share;
	}

	// Set inverse masses using mass
	for (uint32_t i = 0; i < N; ++i) {
		float m = datas_.masses[i];
		if (m > 0.0f)
			datas_.inverse_masses[i] = 1.0f / m;
		else
			datas_.inverse_masses[i] = 0.0f;
	}

	datas_.inverse_masses[0] = 0.0f;
	datas_.inverse_masses[nx1 - 1] = 0.0f;

	// Set stretch edges coloring
	for (int x = 0; x < nx1; ++x)
		for (int y = 0; y + 1 < ny1; y += 2)
			datas_.passes[0].push_back({ vid(x,y), vid(x,y + 1) });

	for (int x = 0; x < nx1; ++x)
		for (int y = 1; y + 1 < ny1; y += 2)
			datas_.passes[1].push_back({ vid(x,y), vid(x,y + 1) });

	for (int y = 0; y < ny1; ++y)
		for (int x = 0; x + 1 < nx1; x += 2)
			datas_.passes[2].push_back({ vid(x,y), vid(x + 1,y) });

	for (int y = 0; y < ny1; ++y)
		for (int x = 1; x + 1 < nx1; x += 2)
			datas_.passes[3].push_back({ vid(x,y), vid(x + 1,y) });

	// Set Edges using coloring
	datas_.pass_offsets[0] = 0;

	for (int p = 0; p < datas_.pass_offsets.size() - 1; ++p) {
		for (auto [i, j] : datas_.passes[p]) {
			glm::vec3 pi = glm::vec3(datas_.positions[i]);
			glm::vec3 pj = glm::vec3(datas_.positions[j]);
			float rest = glm::length(pj - pi);

			datas_.edges.push_back({ i, j, rest, 0.0f });
		}
		datas_.pass_offsets[p + 1] = static_cast<uint32_t>(datas_.edges.size());
	}

	datas_.edge_size_ = static_cast<uint32_t>(datas_.edges.size());
	if (datas_.edge_size_ != ((nx1 - 1) * ny1) + (nx1 * (ny1 - 1)))
	{
		std::cout << datas_.edge_size_ << " " << ((nx1 - 1) * ny1) + (nx1 * (ny1 - 1)) << std::endl;
		throw std::runtime_error("edge size is not right");
	}

	// Set shears
	const size_t numTris = datas_.indices.size() / 3;
	datas_.shears.reserve(numTris);

	for (size_t t = 0; t < numTris; ++t)
	{
		uint32_t i0 = datas_.indices[3 * t + 0];
		uint32_t i1 = datas_.indices[3 * t + 1];
		uint32_t i2 = datas_.indices[3 * t + 2];

		const glm::vec3& x0 = datas_.positions[i0];
		const glm::vec3& x1 = datas_.positions[i1];
		const glm::vec3& x2 = datas_.positions[i2];

		glm::vec3 e1 = x1 - x0;
		glm::vec3 e2 = x2 - x0;

		float restDot = glm::dot(e1, e2);

		SimData::Shear c;
		c.i0 = i0;
		c.i1 = i1;
		c.i2 = i2;
		c.rest_dot = restDot;
		c.lambda = 0.0f;

		datas_.shears.push_back(c);
	}
	datas_.shear_size_ = static_cast<uint32_t>(datas_.shears.size());

	datas_.BuildBendConstraints();
	datas_.bend_size_ = static_cast<uint32_t>(datas_.bends.size());

	datas_.BuildAreaConstraints();
	datas_.area_size_ = static_cast<uint32_t>(datas_.areas.size());

	// position
	ssbo_size_.position = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.position,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.positions,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.position, ssbo_memories_.position,
		&staging_.position, &staging_memories_.position);
	staging_mapped_.position = staging_memories_.position.mapMemory(0, ssbo_size_.position);

	// velocity
	ssbo_size_.velocity = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.velocity,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.velocities,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.velocity, ssbo_memories_.velocity,
		&staging_.velocity, &staging_memories_.velocity);
	staging_mapped_.velocity = staging_memories_.velocity.mapMemory(0, ssbo_size_.velocity);

	// inverse mass
	ssbo_size_.inverse_mass = sizeof(float) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.inverse_mass,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.inverse_masses,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.inverse_mass, ssbo_memories_.inverse_mass,
		&staging_.inverse_mass, &staging_memories_.inverse_mass);
	staging_mapped_.inverse_mass = staging_memories_.inverse_mass.mapMemory(0, ssbo_size_.inverse_mass);

	// delta X
	ssbo_size_.delta_x = sizeof(float) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.delta_x,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		deltaX,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.delta_x, ssbo_memories_.delta_x);

	// delta Y
	ssbo_size_.delta_y = sizeof(float) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.delta_y,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		deltaY,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.delta_y, ssbo_memories_.delta_y);

	// delta Z
	ssbo_size_.delta_z = sizeof(float) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.delta_z,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		deltaZ,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.delta_z, ssbo_memories_.delta_z);

	// delta count
	ssbo_size_.delta_count = sizeof(uint32_t) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.delta_count,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		delta_count,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.delta_count, ssbo_memories_.delta_count);

	// edges
	ssbo_size_.edge = sizeof(SimData::Edge) * datas_.edge_size_;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.edge,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.edges,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.edge, ssbo_memories_.edge,
		&staging_.edge, &staging_memories_.edge);
	staging_mapped_.edge = staging_memories_.edge.mapMemory(0, ssbo_size_.edge);

	// Pred Position
	ssbo_size_.pred_position = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.pred_position,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.pred_positions,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.pred_position, ssbo_memories_.pred_position,
		&staging_.pred_position, &staging_memories_.pred_position);
	staging_mapped_.pred_position = staging_memories_.pred_position.mapMemory(0, ssbo_size_.pred_position);

	// Bend
	ssbo_size_.bend = sizeof(SimData::Bend) * datas_.bend_size_;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.bend,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.bends,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.bend, ssbo_memories_.bend,
		&staging_.bend, &staging_memories_.bend);
	staging_mapped_.bend = staging_memories_.bend.mapMemory(0, ssbo_size_.bend);

	// Shear
	ssbo_size_.shear = sizeof(SimData::Shear) * datas_.shear_size_;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.shear,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.shears,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.shear, ssbo_memories_.shear,
		&staging_.shear, &staging_memories_.shear);
	staging_mapped_.shear = staging_memories_.shear.mapMemory(0, ssbo_size_.shear);

	// grab_counter
	struct GrabState {
		uint32_t id = 0;
		uint32_t dist_bits = 0;
		float	 t = 0.0f;
	};
	std::vector<GrabState> grabState(1);
	ssbo_size_.grab_state = sizeof(GrabState) * grabState.size();
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.grab_state,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		grabState,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.grab_state, ssbo_memories_.grab_state);

	// Area
	ssbo_size_.area = sizeof(SimData::Area) * datas_.area_size_;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.area,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.areas,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.area, ssbo_memories_.area,
		&staging_.area, &staging_memories_.area);
	staging_mapped_.area = staging_memories_.area.mapMemory(0, ssbo_size_.area);
}

void GpuSim::CreateDescriptorSets(Context& context, TextureManager& textureManager)
{
	// Sim Params
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.sim_params
		};

		auto sets = vk::raii::DescriptorSets{ context.device_, allocInfo };
		sets_.sim_params = std::move(sets.front());

		vk::DescriptorBufferInfo simParamsUboInfo{ *ubo_.ubos_.sim_params, 0, sizeof(SimUBO::Data::SimParams) };
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
		context.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Render
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.render
		};

		auto sets = vk::raii::DescriptorSets{ context.device_, allocInfo };
		sets_.render = std::move(sets.front());

		vk::DescriptorBufferInfo renderUboInfo{ *ubo_.ubos_.render, 0, sizeof(SimUBO::Data::Render) };
		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.render,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &renderUboInfo
			}
		};
		context.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Cloth Compute
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.cloth_compute
		};

		auto sets = vk::raii::DescriptorSets{ context.device_, allocInfo };
		sets_.cloth_compute = std::move(sets.front());

		vk::DescriptorBufferInfo positions(*ssbos_.position, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo predPositions(*ssbos_.pred_position, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo velocities(*ssbos_.velocity, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo inverseMass(*ssbos_.inverse_mass, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaX(*ssbos_.delta_x, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaY(*ssbos_.delta_y, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaZ(*ssbos_.delta_z, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaCount(*ssbos_.delta_count, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo edge(*ssbos_.edge, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo shear(*ssbos_.shear, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo bend(*ssbos_.bend, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo grabState(*ssbos_.grab_state, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo area(*ssbos_.area, 0, VK_WHOLE_SIZE);
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
		};
		context.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Cloth Graphics
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.cloth_graphics
		};

		auto sets = vk::raii::DescriptorSets{ context.device_, allocInfo };
		sets_.cloth_graphics = std::move(sets.front());

		vk::DescriptorBufferInfo positions(ssbos_.position, 0, VK_WHOLE_SIZE);

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth_graphics,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &positions
			},

		};
		context.device_.updateDescriptorSets(descriptorWrites, {});
	}
}

void GpuSim::CreateComputePipelines(Context& context)
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
			.size = static_cast<uint32_t>(sizeof(PushConstant::Solve) + sizeof(PushConstant::MouseInteract))
		};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = 2,
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcRange,
		};
		pipeline_layouts_.common = vk::raii::PipelineLayout(context.device_, pipelineLayoutInfo);
	}

	// clear_lambdas
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/clear_lambdas.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.clear_lambdas = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// integrate
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/integrate.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.integrate = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// solve_stretch
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_stretch.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_stretch = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// solve_shear
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_shear.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.solve_shear = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// solve_bend
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_bend.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_bend = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// solve_area
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_area.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_area = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// apply deltas
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/apply_deltas.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.apply_deltas = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// collide_sdf
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/collide_sdf.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.collide_sdf = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// update velocity
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/update_velocity.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common };
		pipelines_.update_velocity = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}
}

void GpuSim::CreateGraphicsPipelines(Context& context, vk::raii::DescriptorSetLayout& globalSetLayout, std::vector<vk::Format>& formats, vk::raii::DescriptorSetLayout& tex2DSetLayout)
{
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
		.topology = vk::PrimitiveTopology::eTriangleList,
		.primitiveRestartEnable = vk::False
	};
	vk::PipelineViewportStateCreateInfo viewportState{
		.viewportCount = 1,
		.scissorCount = 1
	};
	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eNone,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = vk::False
	};
	rasterizer.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisampling{
		.rasterizationSamples = vk::SampleCountFlagBits::e1,
		.sampleShadingEnable = vk::False
	};
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable = vk::True,
		.depthWriteEnable = vk::True,
		.depthCompareOp = vk::CompareOp::eLess,
		.depthBoundsTestEnable = vk::False,
		.stencilTestEnable = vk::False
	};

	std::array<vk::PipelineColorBlendAttachmentState, 3> colorBlendAttachments{};
	for (auto& a : colorBlendAttachments) {
		a.colorWriteMask =
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;
		a.blendEnable = vk::False;
	}

	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable = vk::False,
		.logicOp = vk::LogicOp::eCopy,
		.attachmentCount = colorBlendAttachments.size(),
		.pAttachments = colorBlendAttachments.data()
	};

	std::vector dynamicStates = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor
	};
	vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

	vk::Format depthFormat = vku::FindDepthFormat(context.physical_device_);

	// Cloth
	{
		// Shader
		auto vertCode = vku::ReadFile("shaders/spv/cloth.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/cloth.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context.device_, fragCode);

		vk::PipelineShaderStageCreateInfo vertStage{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = *vertModule,
			.pName = "main"
		};
		vk::PipelineShaderStageCreateInfo fragStage{
			.stage = vk::ShaderStageFlagBits::eFragment,
			.module = *fragModule,
			.pName = "main"
		};
		std::array<vk::PipelineShaderStageCreateInfo, 2> stages{ vertStage, fragStage };

		// SSBO Vertex Pulling
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
			.vertexBindingDescriptionCount = 0,
			.pVertexBindingDescriptions = nullptr,
			.vertexAttributeDescriptionCount = 0,
			.pVertexAttributeDescriptions = nullptr
		};

		vk::PushConstantRange pcRange{
			.stageFlags = vk::ShaderStageFlagBits::eVertex,
			.offset = 0,
			.size = static_cast<uint32_t>(sizeof(PushConstant::ClothRender))
		};
		push_constants_.cloth_render.nx1 = datas_.nx_ + 1;
		push_constants_.cloth_render.ny1 = datas_.ny_ + 1;

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 4> setLayouts(
			*globalSetLayout,
			*set_layouts_.cloth_graphics,
			*set_layouts_.render,
			*tex2DSetLayout);

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = setLayouts.size(),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcRange
		};
		pipeline_layouts_.cloth_graphics = vk::raii::PipelineLayout(context.device_, pipelineLayoutInfo);

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
		  {
			.stageCount = 2,
			.pStages = stages.data(),
			.pVertexInputState = &vertexInputInfo,
			.pInputAssemblyState = &inputAssembly,
			.pViewportState = &viewportState,
			.pRasterizationState = &rasterizer,
			.pMultisampleState = &multisampling,
			.pDepthStencilState = &depthStencil,
			.pColorBlendState = &colorBlending,
			.pDynamicState = &dynamicState,
			.layout = pipeline_layouts_.cloth_graphics,
			.renderPass = nullptr },
		  {.colorAttachmentCount = static_cast<uint32_t>(formats.size()), .pColorAttachmentFormats = formats.data(), .depthAttachmentFormat = depthFormat}
		};
		pipelines_.cloth_solid = vk::raii::Pipeline(context.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rasterizer.polygonMode = vk::PolygonMode::eLine;

		pipelines_.cloth_wireframe = vk::raii::Pipeline(context.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rasterizer.polygonMode = vk::PolygonMode::ePoint;
		pipelines_.cloth_point = vk::raii::Pipeline(context.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}
}