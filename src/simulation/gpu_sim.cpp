#include "context.h"
#include "swapchain.h"
#include "texture.h"
#include "texture_manager.h"
#include "model.h"
#include "model_manager.h"
#include "vulkan_utils.h"

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

void GpuSim::UpdateComputeUBO(uint32_t currentFrame, std::unique_ptr<Model>& model)
{

	ubo_data_.sim_params.dt = 1 / deviding_dt_;
	ubo_data_.sim_params.numParticles = particles_size_;
	ubo_data_.sim_params.numEdges = edge_size;
	ubo_data_.sim_params.numBends = bend_size;
	ubo_data_.sim_params.sphereCenter = glm::vec4(model->position_, 0.0f);
	ubo_data_.sim_params.sphereRadius = model->radius_;

	const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * ubo_size_.sim_params);
	auto* dst = static_cast<std::byte*>(ubo_mapped_.sim_params) + baseOffset;

	std::memcpy(dst, &ubo_data_.sim_params, sizeof(UBOData::SimParams));

}

void GpuSim::ComputeRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::QueryPool& timestampPool, uint32_t& timestampSteps, vku::TestScene& testScene)
{
	cmd.reset();
	cmd.begin({});

	UpdateTestScene(cmd, testScene);

	timestampSteps = 0;
	uint32_t kSlotsPerIterPair = 4 + timestamp_count_ * iterations_ + 2;
	const auto stage = vk::PipelineStageFlagBits2::eComputeShader;
	auto TS = [&](uint32_t& idx) {
		cmd.writeTimestamp2(stage, *timestampPool, idx++);
		};

	cmd.resetQueryPool(*timestampPool, 0, kSlotsPerIterPair);

	uint32_t simparamOffset = currentFrame * static_cast<uint32_t>(ubo_size_.sim_params);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layouts_.common, 0, { sets_.sim_params, sets_.cloth_compute }, { simparamOffset });

	auto ceil_div = [](uint32_t n, uint32_t d) { return (n + d - 1) / d; };
	uint32_t groupsP = ceil_div(particles_size_, 256u);
	uint32_t groupsEdges = ceil_div(edge_size, 256u);

	// 1. Integrate
	TS(timestampSteps);
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.integrate);
	cmd.dispatch(groupsP, 1, 1);
	TS(timestampSteps);
	vku::barrier2(cmd,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	// 2. Clear Lambdas
	TS(timestampSteps);
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.clear_lambdas);
	cmd.dispatch(groupsEdges, 1, 1);
	TS(timestampSteps);
	vku::barrier2(cmd,  // write->read
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	for (uint32_t iter = 0; iter < iterations_; iter++)
	{
		// 3. Solve Coloring - Stretch
		TS(timestampSteps);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_coloring);

		for (uint32_t c = 0; c < 4; ++c) {
			uint32_t base = datas_.pass_offset[c];
			uint32_t count = datas_.pass_offset[c + 1] - datas_.pass_offset[c];
			if (!count) continue;

			pc_.base = base;
			pc_.count = count;
			pc_.compliance = compliance_.stretch;

			cmd.pushConstants<PushConstant>(*pipeline_layouts_.common, vk::ShaderStageFlagBits::eCompute, 0u, pc_);

			uint32_t groups = (count + 256 - 1) / 256;
			cmd.dispatch(groups, 1, 1);
			vku::barrier2(cmd,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
		}
		TS(timestampSteps);

		{
			// 4. Solve AtomicAdd - Diagonal
			TS(timestampSteps);
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_atomic);
			uint32_t base = datas_.pass_offset[4];
			uint32_t count = datas_.pass_offset[5] - datas_.pass_offset[4];
			pc_.base = base;
			pc_.count = count;
			pc_.compliance = compliance_.diagonal;
			cmd.pushConstants<PushConstant>(*pipeline_layouts_.common,
				vk::ShaderStageFlagBits::eCompute, 0u, pc_);

			uint32_t group = (count + 256 - 1) / 256;
			cmd.dispatch(group, 1, 1);
			TS(timestampSteps);
			vku::barrier2(cmd,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
		}

		{
			// 5. Solve Bend
			TS(timestampSteps);
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.solve_bend);
			uint32_t base = 0;
			uint32_t count = bend_size;
			pc_.base = base;
			pc_.count = count;
			pc_.compliance = compliance_.bend;
			cmd.pushConstants<PushConstant>(*pipeline_layouts_.common,
				vk::ShaderStageFlagBits::eCompute, 0u, pc_);

			uint32_t group = (count + 256 - 1) / 256;
			cmd.dispatch(group, 1, 1);
			TS(timestampSteps);
			vku::barrier2(cmd,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
		}

		// 6. Apply Deltas 
		TS(timestampSteps);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.apply_deltas);
		cmd.dispatch(groupsP, 1, 1);
		TS(timestampSteps);
		vku::barrier2(cmd,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageWrite,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

		// 7. Collide SDF
		TS(timestampSteps);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.collide_sdf);
		cmd.dispatch(groupsP, 1, 1);
		TS(timestampSteps);
		vku::barrier2(cmd,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageWrite,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
	}

	// 8. Update Velocity
	TS(timestampSteps);
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines_.update_velocity);
	cmd.dispatch(groupsP, 1, 1);
	TS(timestampSteps);
	vku::barrier2(cmd,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eVertexShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	cmd.end();
}


void GpuSim::UpdateGraphicsUBO(uint32_t currentFrame)
{
	const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * ubo_size_.render);
	auto* dst = static_cast<std::byte*>(ubo_mapped_.render) + baseOffset;

	std::memcpy(dst, &ubo_data_.render, sizeof(UBOData::Render));
}

void GpuSim::GraphicsRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset, vku::PolygonMode mode,
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
		const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * ubo_size_.render);
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

		cmd.pushConstants<ClothPC>(
			*pipeline_layouts_.cloth_graphics,
			vk::ShaderStageFlagBits::eVertex,
			/*offset=*/0,
			cloth_pc_
		);

		cmd.bindIndexBuffer(*index_buffer_, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(indices_size_, 1, 0, 0, 0);
	}
}

void GpuSim::CopyDatas(const vk::raii::CommandBuffer& cmd)
{

	vku::CopyStagingToSSBO(cmd, ssbo_size_.positions, staging_mapped_.positions, datas_.positions, staging_.positions, ssbos_.positions,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.velocities, staging_mapped_.velocities, datas_.velocities, staging_.velocities, ssbos_.velocities,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.inverse_mass, staging_mapped_.inverse_mass, datas_.inverse_mass, staging_.inverse_mass, ssbos_.inverse_mass,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.edges, staging_mapped_.edges, datas_.edges, staging_.edges, ssbos_.edges,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.pred_positions, staging_mapped_.pred_positions, datas_.pred_positions, staging_.pred_positions, ssbos_.pred_positions,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	vku::CopyStagingToSSBO(cmd, ssbo_size_.bends, staging_mapped_.bends, datas_.bends, staging_.bends, ssbos_.bends,
		vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);
}

void GpuSim::UpdateTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene)
{
	spacing_x_ = cloth_size_.x / nx_;
	spacing_y_ = cloth_size_.y / ny_;

	if (testScene.sphereCollision)
	{
		testScene.sphereCollision = false;

		const int nxCells = nx_;
		const int nyCells = ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * nx_) * spacing_x_;   // (-Nx/2 .. +Nx/2) * spacing
				float py = cloth_height_;
				float pz = (y - 0.5f * ny_) * spacing_y_;   // (-Ny/2 .. +Ny/2) * spacing
				datas_.positions[id] = { px, py, pz, 0.0f };
				datas_.velocities[id] = glm::vec4(0.0f);
				float invMass = 1.0f / mass_;
				datas_.inverse_mass[id] = invMass;
				datas_.pred_positions[id] = datas_.positions[id];
			}
		}

		uint32_t idx = 0;
		for (int p = 0; p < 5; ++p) {
			for (auto [i, j] : datas_.pass[p]) {
				glm::vec3 pi = glm::vec3(datas_.positions[i]);
				glm::vec3 pj = glm::vec3(datas_.positions[j]);
				float rest = glm::length(pj - pi);
				datas_.edges[idx++].rest = rest;
			}
		}

		for (int i = 0; i < edge_size; i++)
		{
			datas_.edges[i].lambda = 0.0f;
		}

		for (int i = 0; i < bend_size; i++)
		{
			datas_.bends[i].lambda = 0.0f;
		}

		CopyDatas(cmd);
	}
	else if (testScene.pinnedCorner)
	{
		testScene.pinnedCorner = false;

		const int nxCells = nx_;
		const int nyCells = ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * nx_) * spacing_x_;   // (-Nx/2 .. +Nx/2) * spacing
				float py = cloth_height_;
				float pz = (y - 0.5f * ny_) * spacing_y_;   // (-Ny/2 .. +Ny/2) * spacing
				datas_.positions[id] = { px, py, pz, 0.0f };
				datas_.velocities[id] = glm::vec4(0);
				float invMass = 1.0f / mass_;
				datas_.inverse_mass[id] = invMass;
				datas_.pred_positions[id] = datas_.positions[id];
			}
		}

		datas_.inverse_mass[0] = 0.0f;
		datas_.inverse_mass[nx1 - 1] = 0.0f;
		datas_.inverse_mass[(ny1 - 1) * nx1] = 0.0f;
		datas_.inverse_mass[(ny1 - 1) * nx1 + nx1 - 1] = 0.0f;

		uint32_t idx = 0;
		for (int p = 0; p < 5; ++p) {
			for (auto [i, j] : datas_.pass[p]) {
				glm::vec3 pi = glm::vec3(datas_.positions[i]);
				glm::vec3 pj = glm::vec3(datas_.positions[j]);
				float rest = glm::length(pj - pi);
				datas_.edges[idx++].rest = rest;
			}
		}

		for (int i = 0; i < edge_size; i++)
		{
			datas_.edges[i].lambda = 0.0f;
		}

		for (int i = 0; i < bend_size; i++)
		{
			datas_.bends[i].lambda = 0.0f;
		}

		CopyDatas(cmd);
	}
	else if (testScene.topPinnedCorner)
	{
		testScene.topPinnedCorner = false;

		const int nxCells = nx_;
		const int nyCells = ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * nx_) * spacing_x_;
				float py = cloth_height_;
				float pz = (y - 0.5f * ny_) * spacing_y_;
				datas_.positions[id] = { px, py, pz, 0.0f };
				datas_.velocities[id] = glm::vec4(0);
				float invMass = 1.0f / mass_;
				datas_.inverse_mass[id] = invMass;
				datas_.pred_positions[id] = datas_.positions[id];
			}
		}

		std::vector<uint32_t> indices;
		indices.reserve(nx_ * ny_ * 6);
		for (int y = 0; y < ny_; ++y) {
			for (int x = 0; x < nx_; ++x) {
				uint32_t i0 = vid(x, y);
				uint32_t i1 = vid(x + 1, y);
				uint32_t i2 = vid(x, y + 1);
				uint32_t i3 = vid(x + 1, y + 1);
				indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
				indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
			}
		}

		uint32_t tap = (ny1 - 1) / 10;
		for (uint32_t i = 0; i < ny1; i += tap)
		{
			datas_.inverse_mass[i] = 0.0f;
		}
		//datas_.inverse_mass[(ny1 - 1) * nx1] = 0.0f;
		//datas_.inverse_mass[(ny1 - 1) * nx1 + nx1 - 1] = 0.0f;

		uint32_t idx = 0;
		for (int p = 0; p < 5; ++p) {
			for (auto [i, j] : datas_.pass[p]) {
				glm::vec3 pi = glm::vec3(datas_.positions[i]);
				glm::vec3 pj = glm::vec3(datas_.positions[j]);
				float rest = glm::length(pj - pi);
				datas_.edges[idx++].rest = rest;
			}
		}
		for (int i = 0; i < edge_size; i++)
		{
			datas_.edges[i].lambda = 0.0f;
		}

		for (int i = 0; i < bend_size; i++)
		{
			datas_.bends[i].lambda = 0.0f;
		}

		CopyDatas(cmd);
	}
}

void GpuSim::CreateDescriptorSetLayout(Context& context)
{
	// Sim Params UBO - Compute
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

	// Cloth Compute - Compute
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
		};
		counts_.sb += 10;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.cloth_compute = vk::raii::DescriptorSetLayout(context.device_, layoutInfo);
	}

	// Cloth Rendering - Graphics
	{
		std::array<vk::DescriptorSetLayoutBinding, 1> layoutBindings{
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
	// Sim Params UBO
	{
		ubos_.sim_params.clear();
		ubo_memories_.sim_params.clear();
		ubo_mapped_.sim_params = nullptr;

		auto limits = context.physical_device_.getProperties().limits;
		ubo_size_.sim_params = (sizeof(UBOData::SimParams) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.sim_params * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context.physical_device_, context.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.sim_params = std::move(buffer);
		ubo_memories_.sim_params = std::move(bufferMem);
		ubo_mapped_.sim_params = ubo_memories_.sim_params.mapMemory(0, totalSize);
	}

	// Render UBO
	{
		ubos_.render.clear();
		ubo_memories_.render.clear();
		ubo_mapped_.render = nullptr;

		auto limits = context.physical_device_.getProperties().limits;
		ubo_size_.render = (sizeof(UBOData::Render) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.render * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context.physical_device_, context.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.render = std::move(buffer);
		ubo_memories_.render = std::move(bufferMem);
		ubo_mapped_.render = ubo_memories_.render.mapMemory(0, totalSize);

		ubo_data_.render.albedoEnable = 1;
		ubo_data_.render.albedoIdx = 0;
		//ubo_data_.render.albedoIdx = modelManager.cloth_->texture_idx_.albedo;
		//ubo_data_.render.albedoEnable = modelManager.cloth_->texture_use_.albedo;
		//ubo_data_.render.normalIdx = modelManager.cloth_->texture_idx_.normal;
		//ubo_data_.render.normalEnable = modelManager.cloth_->texture_use_.normal;
		//ubo_data_.render.heightIdx = modelManager.cloth_->texture_idx_.height;
		//ubo_data_.render.heightEnable = modelManager.cloth_->texture_use_.height;
		//ubo_data_.render.roughnessIdx = modelManager.cloth_->texture_idx_.roughness;
		//ubo_data_.render.roughtnessEnable = modelManager.cloth_->texture_use_.roughtness;
	}

}

void GpuSim::CreateSSBOBuffers(Context& context)
{
	const int nxCells = nx_;
	const int nyCells = ny_;
	const int nx1 = nxCells + 1;
	const int ny1 = nyCells + 1;

	auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

	const uint32_t N = nx1 * ny1;
	particles_size_ = N;

	spacing_x_ = cloth_size_.x / nx_;
	spacing_y_ = cloth_size_.y / ny_;

	datas_.positions.resize(N);
	datas_.velocities.resize(N);
	datas_.inverse_mass.resize(N);
	datas_.pred_positions.resize(N);

	for (int y = 0; y < ny1; ++y) {
		for (int x = 0; x < nx1; ++x) {
			uint32_t id = vid(x, y);
			float px = (x - 0.5f * nx_) * spacing_x_;
			float py = cloth_height_;
			float pz = (y - 0.5f * ny_) * spacing_y_;

			datas_.positions[id] = { px, py, pz, 0.0f };
			datas_.velocities[id] = glm::vec4(0);
			datas_.pred_positions[id] = datas_.positions[id];
		}
	}

	std::vector<uint32_t> indices;
	indices.reserve(nx_ * ny_ * 6);
	for (int y = 0; y < ny_; ++y) {
		for (int x = 0; x < nx_; ++x) {
			uint32_t i0 = vid(x, y);
			uint32_t i1 = vid(x + 1, y);
			uint32_t i2 = vid(x, y + 1);
			uint32_t i3 = vid(x + 1, y + 1);
			indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
			indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
		}
	}
	indices_size_ = static_cast<uint32_t>(indices.size());

	std::vector<float> masses(N, 0.0f);

	// Total Area
	float totalArea = 0.0f;
	for (size_t t = 0; t < indices.size(); t += 3) {
		uint32_t i0 = indices[t + 0];
		uint32_t i1 = indices[t + 1];
		uint32_t i2 = indices[t + 2];

		glm::vec3 p0 = glm::vec3(datas_.positions[i0]);
		glm::vec3 p1 = glm::vec3(datas_.positions[i1]);
		glm::vec3 p2 = glm::vec3(datas_.positions[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2)); // Triangle Area
		totalArea += area;
	}

	float totalMassTarget = mass_;

	// area zero defence
	float density = 0.0f;
	if (totalArea > 0.0f) {
		density = totalMassTarget / totalArea; // kg/m©÷
	}

	// Distribute mass to each triangle in proportion to area
	for (size_t t = 0; t < indices.size(); t += 3) {
		uint32_t i0 = indices[t + 0];
		uint32_t i1 = indices[t + 1];
		uint32_t i2 = indices[t + 2];

		glm::vec3 p0 = glm::vec3(datas_.positions[i0]);
		glm::vec3 p1 = glm::vec3(datas_.positions[i1]);
		glm::vec3 p2 = glm::vec3(datas_.positions[i2]);

		glm::vec3 e1 = p1 - p0;
		glm::vec3 e2 = p2 - p0;

		float area = 0.5f * glm::length(glm::cross(e1, e2));

		float triMass = density * area;

		float share = triMass / 3.0f;
		masses[i0] += share;
		masses[i1] += share;
		masses[i2] += share;
	}

	for (uint32_t i = 0; i < N; ++i) {
		float m = masses[i];
		if (m > 0.0f)
			datas_.inverse_mass[i] = 1.0f / m;
		else
			datas_.inverse_mass[i] = 0.0f;
	}

	vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, indices, index_buffer_, index_buffer_memory_);

	std::vector<float > deltaX(N, 0.0f), deltaY(N, 0.0f), deltaZ(N, 0.0f);
	std::vector<uint32_t>  dcount(N, 0);

	for (int x = 0; x < nx1; ++x)
		for (int y = 0; y + 1 < ny1; y += 2)
			datas_.pass[0].push_back({ vid(x,y), vid(x,y + 1) });

	for (int x = 0; x < nx1; ++x)
		for (int y = 1; y + 1 < ny1; y += 2)
			datas_.pass[1].push_back({ vid(x,y), vid(x,y + 1) });

	for (int y = 0; y < ny1; ++y)
		for (int x = 0; x + 1 < nx1; x += 2)
			datas_.pass[2].push_back({ vid(x,y), vid(x + 1,y) });

	for (int y = 0; y < ny1; ++y)
		for (int x = 1; x + 1 < nx1; x += 2)
			datas_.pass[3].push_back({ vid(x,y), vid(x + 1,y) });

	for (int y = 0; y + 1 < ny1; ++y)
		for (int x = 0; x + 1 < nx1; ++x) {
			datas_.pass[4].push_back({ vid(x,y),     vid(x + 1,y + 1) }); // "\"
			datas_.pass[4].push_back({ vid(x + 1,y),   vid(x,  y + 1) }); // "/"
		}

	datas_.pass_offset[0] = 0;

	for (int p = 0; p < 5; ++p) {
		for (auto [i, j] : datas_.pass[p]) {
			glm::vec3 pi = glm::vec3(datas_.positions[i]);
			glm::vec3 pj = glm::vec3(datas_.positions[j]);
			float rest = glm::length(pj - pi);
			datas_.edges.push_back({ i, j, rest, 0.0f });
		}
		datas_.pass_offset[p + 1] = static_cast<uint32_t>(datas_.edges.size());
	}

	for (int y = 0; y < ny_; ++y) {
		for (int x = 0; x < nx_; ++x) {
			uint32_t i0 = vid(x, y);
			uint32_t i1 = vid(x + 1, y);
			uint32_t i2 = vid(x, y + 1);
			uint32_t i3 = vid(x + 1, y + 1);

			uint32_t p1 = i1;   // hinge start
			uint32_t p2 = i2;   // hinge end
			uint32_t p3 = i0;   // opp of first tri
			uint32_t p4 = i3;   // opp of second tri

			float theta0 = 0.0f;
			//theta0 = ComputeRestAngle(p1,p2,p3,p4, data_.positions);

			datas_.bends.push_back({ p1,p2,p3,p4,theta0, 0.0f, glm::vec2(0.0f, 0.0f) });
		}
	}
	bend_size = static_cast<uint32_t>(datas_.bends.size());

	edge_size = static_cast<uint32_t>(datas_.edges.size());

	datas_.sdfColliders.push_back(Data::SDFCollider{ 0, glm::vec3(0.0f, 0.0f, 0.0f), 1.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), 0.0f });

	// position
	ssbo_size_.positions = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.positions,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.positions,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.positions, ssbo_memories_.positions,
		&staging_.positions, &staging_memories_.positions);
	staging_mapped_.positions = staging_memories_.positions.mapMemory(0, ssbo_size_.positions);

	// velocity
	ssbo_size_.velocities = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.velocities,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.velocities,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.velocities, ssbo_memories_.velocities,
		&staging_.velocities, &staging_memories_.velocities);
	staging_mapped_.velocities = staging_memories_.velocities.mapMemory(0, ssbo_size_.velocities);

	// inverse mass
	ssbo_size_.inverse_mass = sizeof(float) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.inverse_mass,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.inverse_mass,
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
	ssbo_size_.dcount = sizeof(uint32_t) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.dcount,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		dcount,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.dcount, ssbo_memories_.dcount);

	// edges
	ssbo_size_.edges = sizeof(Data::Edge) * edge_size;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.edges,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.edges,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.edges, ssbo_memories_.edges,
		&staging_.edges, &staging_memories_.edges);
	staging_mapped_.edges = staging_memories_.edges.mapMemory(0, ssbo_size_.edges);

	// Pred Position
	ssbo_size_.pred_positions = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.pred_positions,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.pred_positions,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.pred_positions, ssbo_memories_.pred_positions,
		&staging_.pred_positions, &staging_memories_.pred_positions);
	staging_mapped_.pred_positions = staging_memories_.pred_positions.mapMemory(0, ssbo_size_.pred_positions);

	// Data::Bend
	ssbo_size_.bends = sizeof(Data::Bend) * bend_size;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		ssbo_size_.bends,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.bends,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		ssbos_.bends, ssbo_memories_.bends,
		&staging_.bends, &staging_memories_.bends);
	staging_mapped_.bends = staging_memories_.bends.mapMemory(0, ssbo_size_.bends);

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

		vk::DescriptorBufferInfo simParamsUboInfo{ *ubos_.sim_params, 0, sizeof(UBOData::SimParams) };
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

		vk::DescriptorBufferInfo renderUboInfo{ *ubos_.render, 0, sizeof(UBOData::Render) };
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

		vk::DescriptorBufferInfo positions(*ssbos_.positions, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo velocities(*ssbos_.velocities, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo inverseMass(*ssbos_.inverse_mass, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaX(*ssbos_.delta_x, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaY(*ssbos_.delta_y, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo deltaZ(*ssbos_.delta_z, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo dcount(*ssbos_.dcount, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo edge(*ssbos_.edges, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo predPositions(*ssbos_.pred_positions, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo bend(*ssbos_.bends, 0, VK_WHOLE_SIZE);
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
				.pBufferInfo = &velocities
			},
				vk::WriteDescriptorSet{
					.dstSet = *sets_.cloth_compute,
					.dstBinding = 2,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &inverseMass
			},
				vk::WriteDescriptorSet{
					.dstSet = *sets_.cloth_compute,
					.dstBinding = 3,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &deltaX
			},
				vk::WriteDescriptorSet{
					.dstSet = *sets_.cloth_compute,
					.dstBinding = 4,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &deltaY
			},
				vk::WriteDescriptorSet{
					.dstSet = *sets_.cloth_compute,
					.dstBinding = 5,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &deltaZ
			},
				vk::WriteDescriptorSet{
					.dstSet = *sets_.cloth_compute,
					.dstBinding = 6,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &dcount
			},
				vk::WriteDescriptorSet{
					.dstSet = *sets_.cloth_compute,
					.dstBinding = 7,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &edge
			},
				vk::WriteDescriptorSet{
					.dstSet = *sets_.cloth_compute,
					.dstBinding = 8,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &predPositions
			},
				vk::WriteDescriptorSet{
					.dstSet = *sets_.cloth_compute,
					.dstBinding = 9,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &bend
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

		vk::DescriptorBufferInfo positions(ssbos_.positions, 0, VK_WHOLE_SIZE);

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
		std::array<vk::DescriptorSetLayout, 2> setLayouts(*set_layouts_.sim_params, *set_layouts_.cloth_compute);

		vk::PushConstantRange pcRange{
			.stageFlags = vk::ShaderStageFlagBits::eCompute,
			.offset = 0,
			.size = static_cast<uint32_t>(sizeof(PushConstant))
		};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = 2,
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcRange
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

	// solve_coloring
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_coloring.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_coloring = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// solve_atomic
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_atomic.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_atomic = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
	}

	// solve_bend
	{
		vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_bend.comp.spv"));

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

		vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *pipeline_layouts_.common,
		};
		pipelines_.solve_bend = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
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
			.size = static_cast<uint32_t>(sizeof(ClothPC))
		};
		cloth_pc_.nx1 = nx_ + 1;
		cloth_pc_.ny1 = ny_ + 1;

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
