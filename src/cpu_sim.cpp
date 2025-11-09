#include "texture_2d.h"
#include "swapchain.h"
#include "vulkan_utils.h"

#include "cpu_sim.h"

CpuSim::CpuSim(
	vk::raii::PhysicalDevice& physicalDevice, 
	vk::raii::Device& device, 
	vk::raii::Queue& queue,
	vk::raii::CommandPool& commandPool,
	std::unique_ptr<Swapchain>& swapchain,
	uint32_t Nx, uint32_t Ny, float spacing, std::unique_ptr<Texture2D>& texture,
	vk::raii::DescriptorSetLayout& globalSetLayout)
{
	Nx_ = Nx;
	Ny_ = Ny;
	particles_size_ = Nx_ * Ny_;
	spacing_ = spacing;

	{
		CreateClothData_CPU(physicalDevice, device, queue, commandPool);
	}

	// Descriptor set layout
	{
		std::array<vk::DescriptorSetLayoutBinding, 2> layoutBindings{
			vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex },
			vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
		};
		counts_.sb += 1 * MAX_FRAMES_IN_FLIGHT;
		counts_.sampler += 1 * MAX_FRAMES_IN_FLIGHT;
		counts_.layout += 1 * MAX_FRAMES_IN_FLIGHT;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		sim_cpu_descriptor_set_layout_ = vk::raii::DescriptorSetLayout(device, layoutInfo);

	}

	// Descriptor Pool
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

		descriptor_pool_ = vk::raii::DescriptorPool(device, poolInfo);
	}

	// Descriptor Set
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, sim_cpu_descriptor_set_layout_);
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
			.pSetLayouts = layouts.data()
		};

		sim_cpu_descriptor_set_.clear();
		sim_cpu_descriptor_set_ = vk::raii::DescriptorSets{ device, allocInfo };

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DescriptorBufferInfo positions(pos_ssbo_[i], 0, VK_WHOLE_SIZE);
			vk::DescriptorImageInfo imageInfo{
				.sampler = *texture->texture_sampler_,
				.imageView = *texture->texture_image_view_,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			};
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = *sim_cpu_descriptor_set_[i],
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &positions
				},
				vk::WriteDescriptorSet{
					.dstSet = *sim_cpu_descriptor_set_[i],
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eCombinedImageSampler,
					.pImageInfo = &imageInfo
				}
			};

			device.updateDescriptorSets(descriptorWrites, {});
		}
	}

	// pipeline
	{
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
				.cullMode = vk::CullModeFlagBits::eBack,
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
			vk::PipelineColorBlendAttachmentState colorBlendAttachment;
			colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
			colorBlendAttachment.blendEnable = vk::False;

			vk::PipelineColorBlendStateCreateInfo colorBlending{
				.logicOpEnable = vk::False,
				.logicOp = vk::LogicOp::eCopy,
				.attachmentCount = 1,
				.pAttachments = &colorBlendAttachment
			};

			std::vector dynamicStates = {
				vk::DynamicState::eViewport,
				vk::DynamicState::eScissor
			};
			vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

			vk::Format depthFormat = vku::FindDepthFormat(physicalDevice);

			// Shader
			auto vertCode = vku::ReadFile("shaders/spv/cloth.vert.spv");
			auto fragCode = vku::ReadFile("shaders/spv/cloth.frag.spv");

			vk::raii::ShaderModule vertModule = vku::CreateShaderModule(device, vertCode);
			vk::raii::ShaderModule fragModule = vku::CreateShaderModule(device, fragCode);

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

			// push constant 범위: VS에서만 사용(필요하면 FS도 추가)
			vk::PushConstantRange pcRange{
				.stageFlags = vk::ShaderStageFlagBits::eVertex,
				.offset = 0,
				.size = static_cast<uint32_t>(sizeof(ClothPC))
			};

			// Pipeline Layout
			std::array<vk::DescriptorSetLayout, 2> setLayouts(*globalSetLayout, *sim_cpu_descriptor_set_layout_);
			vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
				.setLayoutCount = 2,
				.pSetLayouts = setLayouts.data(),
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &pcRange
			};
			sim_cpu_pipeline_layout_ = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

			rasterizer.cullMode = vk::CullModeFlagBits::eNone;

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
				.layout = sim_cpu_pipeline_layout_,
				.renderPass = nullptr },
			  {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchain->swapchain_surface_format_.format, .depthAttachmentFormat = depthFormat }
			};
			sim_cpu_pipeline_ = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		}
	}
}

void CpuSim::CreateClothData_CPU(
	vk::raii::PhysicalDevice& physicalDevice, 
	vk::raii::Device& device,
	vk::raii::Queue& queue,
	vk::raii::CommandPool& commandPool)
{
	particles_size_ = Nx_ * Ny_;
	const int N = particles_size_;
	positions.resize(N);
	velocities.resize(N, glm::vec4(0.0f));
	invMass.resize(N, 1.0f);

	const float width = spacing_ * (Nx_ - 1);
	const float height = spacing_ * (Ny_ - 1);

	const float originX = -width * 0.5f;
	const float originY = -height * 0.5f;
	const float zPlane = 0.0f;

	for (int y = 0; y < Ny_; ++y)
		for (int x0 = 0; x0 < Nx_; ++x0) {
			int id = y * Nx_ + x0;

			float px = originX + x0 * spacing_;
			float py = originY + y * spacing_;
			float pz = zPlane;
			glm::vec3 pos(originX + x0 * spacing_, 2.0f, -originY - y * spacing_);
			positions[id] = glm::vec4(pos, 1.0f);
		}


	auto idx = [&](int x, int y) { return y * Nx_ + x; };

	indices_size_ = (Nx_ - 1) * (Ny_ - 1) * 6;

	for (int y = 0; y < Ny_ - 1; ++y) {
		for (int x = 0; x < Nx_ - 1; ++x) {
			uint32_t i0 = idx(x, y);
			uint32_t i1 = idx(x + 1, y);
			uint32_t i2 = idx(x, y + 1);
			uint32_t i3 = idx(x + 1, y + 1);

			indices_cpu.push_back(i1); indices_cpu.push_back(i2); indices_cpu.push_back(i0);
			indices_cpu.push_back(i1); indices_cpu.push_back(i3); indices_cpu.push_back(i2);
		}
	}
	vku::CreateIndexBuffer(physicalDevice, device, queue, commandPool, indices_cpu, ib, ibm);

	VkDeviceSize posSize = sizeof(glm::vec4) * particles_size_;
	pos_ssbo_.reserve(MAX_FRAMES_IN_FLIGHT);
	pos_ssbo_mem_.reserve(MAX_FRAMES_IN_FLIGHT);
	pos_staging_.reserve(MAX_FRAMES_IN_FLIGHT);
	pos_staging_mem_.reserve(MAX_FRAMES_IN_FLIGHT);
	pos_staging_map_.resize(MAX_FRAMES_IN_FLIGHT);

	for (uint32_t k = 0; k < MAX_FRAMES_IN_FLIGHT; ++k) {
		// Staging (HOST_VISIBLE | HOST_COHERENT, TRANSFER_SRC)
		vk::raii::Buffer stagingBuf{ nullptr };
		vk::raii::DeviceMemory stagingMem{ nullptr };
		vk::raii::Buffer devBuf{ nullptr };
		vk::raii::DeviceMemory devMem{ nullptr };

		vku::CreateBuffer(
			physicalDevice, device, posSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuf, stagingMem
		);

		// Device-local target (STORAGE | TRANSFER_DST)
		vku::CreateBuffer(
			physicalDevice, device, posSize,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			devBuf, devMem
		);

		pos_staging_.push_back(std::move(stagingBuf));
		pos_staging_mem_.push_back(std::move(stagingMem));
		pos_ssbo_.push_back(std::move(devBuf));
		pos_ssbo_mem_.push_back(std::move(devMem));

		pos_staging_map_[k] = pos_staging_mem_[k].mapMemory(0, posSize);
	}

	// 2) 초기 값 업로드(한 번)
	std::memcpy(pos_staging_map_[0], positions.data(), (size_t)posSize);
	// 초기 카피는 임의 프레임 k로 실행해도 됨
	vku::CopyBuffer(device, queue, commandPool, pos_staging_[0], pos_ssbo_[0], posSize);

	// --- top row pin ---
	for (int x0 = 0; x0 < Nx_; ++x0)
		invMass[(Ny_ - 1) * Nx_ + x0] = 0.0f;

	// --- edges (structural only) ---
	edges.clear();

	for (int y = 0; y < Ny_; ++y) {
		for (int x0 = 0; x0 < Nx_; ++x0) {
			uint32_t i = idx(x0, y);
			if (x0 < Nx_ - 1) { // horizontal
				uint32_t j = idx(x0 + 1, y);
				edges.push_back({ i, j, spacing_, 0.0f });
			}
			if (y < Ny_ - 1) { // vertical
				uint32_t j = idx(x0, y + 1);
				edges.push_back({ i, j, spacing_, 0.0f });
			}
		}
	}

	// --- optional shear / bend edges (for realism) ---
	for (uint32_t y = 0; y < Ny_ - 1; ++y) {
		for (uint32_t x0 = 0; x0 < Nx_ - 1; ++x0) {
			uint32_t i = idx(x0, y);
			edges.push_back({ i, idx(x0 + 1, y + 1), spacing_ * 1.414f, 0.0f });
			edges.push_back({ idx(x0 + 1, y), idx(x0, y + 1), spacing_ * 1.414f, 0.0f });
		}
	}
}

void CpuSim::SimulateClothXPBD_CPU(
	const glm::vec3& sphereCenter,
	float sphereRadius
) {
	const size_t N = particles_size_;
	std::vector<glm::vec4> xp(N);

	// 1. integrate
	for (size_t i = 0; i < N; ++i) {
		if (invMass[i] == 0.0f) { xp[i] = positions[i]; continue; }
		glm::vec3 xi = glm::vec3(positions[i]);
		glm::vec3 vi = glm::vec3(velocities[i]);
		vi += gravity_ * dt_;
		xi += vi * dt_;
		xp[i] = glm::vec4(xi, 1.0f);
		velocities[i] = glm::vec4(vi, 0.0f);
	}

	for (auto& e : edges) {
		e.lambda = 0.0f;
	}

	// 2. XPBD 반복
	for (int iter = 0; iter < iterations_; ++iter) {
		// 거리 제약
		for (auto& e : edges) {
			uint32_t i = e.i, j = e.j;
			float wi = invMass[i], wj = invMass[j];
			if (wi + wj < 1e-8f) continue;

			glm::vec3 xi = glm::vec3(xp[i]);
			glm::vec3 xj = glm::vec3(xp[j]);
			glm::vec3 n = xi - xj;
			float L = glm::length(n);
			if (L < 1e-8f) continue;
			n /= L;

			float C = L - e.rest;
			float alpha = compliance / (dt_ * dt_);
			float denom = wi + wj + alpha;
			float dl = -(C + alpha * e.lambda) / denom;
			e.lambda += dl;

			glm::vec3 corr = n * dl;
			if (wi > 0.0f) xp[i] += glm::vec4(wi * corr, 0.0f);
			if (wj > 0.0f) xp[j] -= glm::vec4(wj * corr, 0.0f);
		}

		// 구 충돌 (투영)
		for (size_t i = 0; i < N; ++i) {
			if (invMass[i] == 0.0f) continue;
			glm::vec3 p = glm::vec3(xp[i]);
			glm::vec3 d = p - sphereCenter;
			float dist = glm::length(d);
			if (dist < sphereRadius) {
				glm::vec3 n = (dist > 1e-8f) ? (d / dist) : glm::vec3(0, 1, 0);
				float beta = 0.3f; // 부분 보정
				p += (sphereRadius - dist) * beta * n;
				xp[i] = glm::vec4(p, 1.0f);
			}
		}
	}

	// 3. 속도 업데이트
	for (size_t i = 0; i < N; ++i) {
		if (invMass[i] == 0.0f) continue;
		glm::vec3 newV = (glm::vec3(xp[i]) - glm::vec3(positions[i])) / dt_;
		newV *= (1.0f - damping);
		velocities[i] = glm::vec4(newV, 0.0f);
		positions[i] = xp[i];
	}
}

void CpuSim::CopyPositions(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd)
{
	VkDeviceSize size = sizeof(glm::vec4) * particles_size_;

	// 1) staging memcpy
	std::memcpy(pos_staging_map_[currentFrame], positions.data(), (size_t)size);
	// HostCoherent가 아니면 vkFlushMappedMemoryRanges 호출

	// 2) copy staging -> device
	vk::BufferCopy region{ 0, 0, size };
	cmd.copyBuffer(*pos_staging_[currentFrame], *pos_ssbo_[currentFrame], { region });

	// 3) barrier: TRANSFER_WRITE -> VERTEX/SSBO read
	vk::BufferMemoryBarrier2 b{
		.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
		.srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
		// SSBO를 VS에서 읽는다면:
		.dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
		.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
		.buffer = *pos_ssbo_[currentFrame],
		.offset = 0,
		.size = size
	};
	vk::DependencyInfo dep{
		.bufferMemoryBarrierCount = 1,
		.pBufferMemoryBarriers = &b
	};
	cmd.pipelineBarrier2(dep);

}

void CpuSim::UpdatePushContants()
{
	cloth_pc_.Nx = Nx_;
	cloth_pc_.Ny = Ny_;
}

void CpuSim::Record(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset)
{
	UpdatePushContants();

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *sim_cpu_pipeline_);

	// Global Set
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		sim_cpu_pipeline_layout_,
		0,
		{ *globalSet },
		{ globalOffset }
	);
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		sim_cpu_pipeline_layout_,
		1,
		{ *sim_cpu_descriptor_set_[currentFrame] },
		{}
	);

	cmd.pushConstants<ClothPC>(
		*sim_cpu_pipeline_layout_,
		vk::ShaderStageFlagBits::eVertex,
		/*offset=*/0,
		cloth_pc_
	);

	cmd.bindIndexBuffer(*ib, 0, vk::IndexType::eUint32);
	cmd.drawIndexed(indices_size_, 1, 0, 0, 0);
}