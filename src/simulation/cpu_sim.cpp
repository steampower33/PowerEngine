#include "context.h"
#include "swapchain.h"
#include "model_manager.h"
#include "texture.h"
#include "texture_manager.h"
#include "vulkan_utils.h"
#include "ray.h"
#include "mouse_interactor.h"

#include "cpu_sim.h"

CpuSim::CpuSim(
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
	CreateDatas(context);
	CreateDescriptorSets(context, textureManager);
	CreateGraphicsPipelines(context, globalSetLayout, formats, tex2DSetLayout);
}

void CpuSim::UpdateMousePushConstant(Camera& camera, MouseInteractor& mouseInteractor, glm::vec2 viewportSize)
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

void CpuSim::ComputeSolve(const glm::vec3& sphereCenter, float sphereRadius)
{
	auto& d = datas_;

	const uint32_t N = d.particles_size_;
	if (N == 0) return;

	const float frame_dt = 1.0f / d.frame_dt_;
	const int   substeps = std::max(1, d.substeps_);
	const float dt = frame_dt / static_cast<float>(substeps);

	const int iterations = std::max(1, d.iterations_);

	const glm::vec3 gravity(0.0f, -9.81f, 0.0f);
	const float velocity_damping = 0.01f;

	auto& pc = push_constants_.mouse_interact;

	auto mouse_select = [&](int i, glm::vec3& pos) {
		if (pc.select_mode == 0)
		{
			id = -1;
			dist2 = 1000.0f;
			T = 1000.0f;
		}

		if (pc.select_mode != 1)
			return;

		glm::vec3 o = pc.ray_origin;
		glm::vec3 dir = pc.ray_dir;

		glm::vec3 v = pos - o;
		float t = std::max(dot(v, dir), 0.0f);
		glm::vec3 c = o + dir * t;
		float d2 = dot(pos - c, pos - c);

		float r = pc.radius;
		if (d2 > r * r) return;

		if (dist2 > d2)
		{
			id = i;
			dist2 = d2;
			T = t;
			//std::cout << "[Select] id : " << id << ", dist2 : " << dist2 << ", T : " << T << std::endl;
		}

		};

	auto mouse_drag = [&](int i, glm::vec3& xpi, glm::vec3& vi) {
		if (pc.select_mode != 2)
			return;

		if (dist2 >= 999.9f)
			return;

		if (i != id)
			return;

		glm::vec3 o = pc.ray_origin;
		glm::vec3 dir = pc.ray_dir;

		float t = T;
		glm::vec3 c = o + t * dir;

		xpi = c;
		vi = glm::vec3(0.0f);

		//std::cout << "[Drag] id : " << id << ", dist2 : " << dist2 << ", T : " << T << std::endl;
		};

	for (int step = 0; step < substeps; ++step)
	{
		for (uint32_t i = 0; i < N; ++i)
		{
			float w = d.inverse_masses[i];
			glm::vec3 x = glm::vec3(d.positions[i]);
			glm::vec3 v = glm::vec3(d.velocities[i]);

			if (w == 0.0f) {
				d.pred_positions[i] = d.positions[i];
				continue;
			}

			mouse_select(i, x);

			v += gravity * dt;

			glm::vec3 xpi = x + v * dt;

			mouse_drag(i, xpi, v);

			d.pred_positions[i] = glm::vec4(xpi, 0.0f);
			d.velocities[i] = glm::vec4(v, 0.0f);
		}

		for (auto& e : d.edges) {
			e.lambda = 0.0f;
		}

		for (int iter = 0; iter < iterations; ++iter)
		{
			const float alpha_stretch = d.compliance_.stretch / (dt * dt);

			for (auto& e : d.edges)
			{
				uint32_t i = e.i;
				uint32_t j = e.j;

				float wi = d.inverse_masses[i];
				float wj = d.inverse_masses[j];
				if (wi + wj <= 0.0f)
					continue;

				glm::vec3 xi = glm::vec3(d.pred_positions[i]);
				glm::vec3 xj = glm::vec3(d.pred_positions[j]);

				glm::vec3 n = xi - xj;
				float L = glm::length(n);
				if (L < 1e-8f)
					continue;

				n /= L;

				float C = L - e.rest;

				float denom = wi + wj + alpha_stretch;
				float dl = -(C + alpha_stretch * e.lambda) / denom;
				e.lambda += dl;

				glm::vec3 corr = dl * n;

				if (wi > 0.0f)
					d.pred_positions[i] += glm::vec4(wi * corr, 0.0f);
				if (wj > 0.0f)
					d.pred_positions[j] -= glm::vec4(wj * corr, 0.0f);
			}

			for (uint32_t i = 0; i < N; ++i)
			{
				float w = d.inverse_masses[i];
				if (w == 0.0f) continue;

				glm::vec3 p = glm::vec3(d.pred_positions[i]);
				glm::vec3 dvec = p - sphereCenter;
				float dist = glm::length(dvec);

				if (dist < sphereRadius)
				{
					glm::vec3 n = (dist > 1e-8f) ? (dvec / dist) : glm::vec3(0, 1, 0);
					p = sphereCenter + n * sphereRadius;
					d.pred_positions[i] = glm::vec4(p, 1.0f);
				}
			}

			// shear / bend / area
		}

		for (uint32_t i = 0; i < N; ++i)
		{
			float w = d.inverse_masses[i];
			glm::vec3 x_old = glm::vec3(d.positions[i]);
			glm::vec3 x_new = glm::vec3(d.pred_positions[i]);

			if (w == 0.0f) {
				d.positions[i] = glm::vec4(x_old, 1.0f);
				d.velocities[i] = glm::vec4(0.0f);
				continue;
			}

			glm::vec3 v = (x_new - x_old) / dt;
			v *= (1.0f - velocity_damping);

			d.positions[i] = glm::vec4(x_new, 1.0f);
			d.velocities[i] = glm::vec4(v, 0.0f);
		}
	}
}

void CpuSim::CopyPositions(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd)
{
	vku::CopyStagingToSSBO(cmd, pos_ssbo_size_, pos_staging_map_, datas_.positions, pos_staging_, pos_ssbo_, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderStorageRead);
}

void CpuSim::UpdateGraphicsUBO(uint32_t currentFrame)
{
	const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * ubo_.size_.render);
	auto* dst = static_cast<std::byte*>(ubo_.mapped_.render) + baseOffset;

	std::memcpy(dst, &ubo_.datas_.render, sizeof(SimUBO::Data::Render));
}

void CpuSim::RecordGraphics(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset, vku::PolygonMode mode,
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

void CpuSim::CreateDescriptorSetLayout(Context& context)
{
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


void CpuSim::CreateDescriptorPools(Context& context)
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

void CpuSim::CreateUniformBuffers(Context& context,
	ModelManager& modelManager)
{
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

void CpuSim::CreateDatas(Context& context)
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
	pos_ssbo_size_ = sizeof(glm::vec4) * N;
	vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
		pos_ssbo_size_,
		vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		datas_.positions,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		pos_ssbo_, pos_ssbo_mem_,
		&pos_staging_, &pos_staging_mem_);
	pos_staging_map_ = pos_staging_mem_.mapMemory(0, pos_ssbo_size_);
}

void CpuSim::CreateDescriptorSets(Context& context, TextureManager& textureManager)
{
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

	// Cloth Graphics
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.cloth_graphics
		};

		auto sets = vk::raii::DescriptorSets{ context.device_, allocInfo };
		sets_.cloth_graphics = std::move(sets.front());

		vk::DescriptorBufferInfo positions(pos_ssbo_, 0, VK_WHOLE_SIZE);

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

void CpuSim::CreateGraphicsPipelines(Context& context, vk::raii::DescriptorSetLayout& globalSetLayout, std::vector<vk::Format>& formats, vk::raii::DescriptorSetLayout& tex2DSetLayout)
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