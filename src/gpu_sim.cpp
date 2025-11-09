#include "texture_2d.h"
#include "swapchain.h"
#include "vulkan_utils.h"

#include "gpu_sim.h"

GpuSim::GpuSim(
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
	spacing_ = spacing;
	particles_size_ = Nx_ * Ny_;

	// Command Buffer
	{
		compute_.command_buffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.commandPool = *commandPool;
		allocInfo.level = vk::CommandBufferLevel::ePrimary;
		allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
		compute_.command_buffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	// Descriptor Set Layout
	{
		// Sim Params UBO - Compute
		{
			std::array layoutBindings{
				vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eCompute, nullptr)
			};
			counts_.ubo_dynamic += 1;
			counts_.layout += 1;

			vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
			compute_.sim_params_set_layout = vk::raii::DescriptorSetLayout(device, layoutInfo);
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
			};
			counts_.sb += 8;
			counts_.layout += 1;

			vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
			compute_.cloth_compute_set_layout = vk::raii::DescriptorSetLayout(device, layoutInfo);
		}

		// Cloth Rendering - Graphics
		{
			std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{
				vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex },
				vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
				vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex }
			};
			counts_.ubo_dynamic += 1;
			counts_.sampler += 1;
			counts_.sb += 1;
			counts_.layout += 1;

			vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
			graphics_.cloth_set_layout = vk::raii::DescriptorSetLayout(device, layoutInfo);
		}
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

	// Sim Params UBO
	{
		compute_.sim_params_ubo.clear();
		compute_.sim_params_ubo_memory.clear();
		compute_.sim_params_ubo_mapped = nullptr;

		auto limits = physicalDevice.getProperties().limits;
		compute_.sim_params_slot_size = (sizeof(Compute::SimParams) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = compute_.sim_params_slot_size * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(physicalDevice, device, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		compute_.sim_params_ubo = std::move(buffer);
		compute_.sim_params_ubo_memory = std::move(bufferMem);
		compute_.sim_params_ubo_mapped = compute_.sim_params_ubo_memory.mapMemory(0, totalSize);
	}

	// SSBOs
	{
		const float width = spacing_ * (Nx_ - 1);
		const float height = spacing_ * (Ny_ - 1);

		const float originX = -width * 0.5f;
		const float originY = -height * 0.5f;
		const float zPlane = 0.0f;

		auto idx = [&](int x, int y) { return y * Nx_ + x; };

		// 입력
		std::vector<glm::vec3> positions(particles_size_);

		for (int y = 0; y < Ny_; ++y)
			for (int x = 0; x < Nx_; ++x) {
				const int id = idx(x, y);

				float px = originX + x * spacing_;
				float py = originY + y * spacing_;
				float pz = zPlane;

				positions[id] = { originX + x * spacing_, 2.0f, -originY - y * spacing_ };
			}

		indices_size_ = (Nx_ - 1) * (Ny_ - 1) * 6;
		std::vector<uint32_t> indices;
		std::vector<glm::uvec3> triangles;

		for (int y = 0; y < Ny_ - 1; ++y) {
			for (int x = 0; x < Nx_ - 1; ++x) {
				uint32_t i0 = idx(x, y);
				uint32_t i1 = idx(x + 1, y);
				uint32_t i2 = idx(x, y + 1);
				uint32_t i3 = idx(x + 1, y + 1);

				indices.push_back(i1); indices.push_back(i2); indices.push_back(i0);
				triangles.push_back({ i1, i2, i0 });
				indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
				triangles.push_back({ i1, i3, i2 });
			}
		}
		vku::CreateIndexBuffer(physicalDevice, device, queue, commandPool, indices, index_buffer_, index_buffer_memory_);

		// 출력 SoA
		const uint32_t V = (uint32_t)positions.size();
		std::vector<glm::vec4> x(V), v(V, glm::vec4(0.0f));
		std::vector<float>     w(V, 1.0f);
		std::vector<uint32_t> deltaX(V, 0), deltaY(V, 0), deltaZ(V, 0);
		std::vector<uint32_t>  dcount(V, 0);
		std::vector<Edge> edge(V);

		for (uint32_t i = 0; i < V; ++i) {
			x[i] = glm::vec4(positions[i], 1.0f);
		}

		// --- top row pin ---
		for (int x0 = 0; x0 < Nx_; ++x0)
			w[(Ny_ - 1) * Nx_ + x0] = 0.0f;

		// --- edge (structural only) ---
		edge.clear();

		for (int y = 0; y < Ny_; ++y) {
			for (int x0 = 0; x0 < Nx_; ++x0) {
				uint32_t i = idx(x0, y);
				if (x0 < Nx_ - 1) { // horizontal
					uint32_t j = idx(x0 + 1, y);
					edge.push_back({ i, j, spacing_, 0.0f });
				}
				if (y < Ny_ - 1) { // vertical
					uint32_t j = idx(x0, y + 1);
					edge.push_back({ i, j, spacing_, 0.0f });
				}
			}
		}

		// --- optional shear / bend edge (for realism) ---
		for (uint32_t y = 0; y < Ny_ - 1; ++y) {
			for (uint32_t x0 = 0; x0 < Nx_ - 1; ++x0) {
				uint32_t i = idx(x0, y);
				edge.push_back({ i, idx(x0 + 1, y + 1), spacing_ * 1.414f, 0.0f });
				edge.push_back({ idx(x0 + 1, y), idx(x0, y + 1), spacing_ * 1.414f, 0.0f });
			}
		}

		// position
		positions_ssbo_size_ = sizeof(glm::vec4) * particles_size_;
		vku::CreateSSBO(physicalDevice, device, queue, commandPool,
			positions_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			x,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			positions_ssbo_, positions_ssbo_memory_);

		// velocity
		velocities_ssbo_size_ = sizeof(glm::vec4) * particles_size_;
		vku::CreateSSBO(physicalDevice, device, queue, commandPool,
			velocities_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			v,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			velocities_ssbo_, velocities_ssbo_memory_);

		// inverse mass
		inverse_mass_ssbo_size_ = sizeof(float) * particles_size_;
		vku::CreateSSBO(physicalDevice, device, queue, commandPool,
			inverse_mass_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			w,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			inverse_mass_ssbo_, inverse_mass_ssbo_memory_);

		// delta X
		delta_x_ssbo_size_ = sizeof(uint32_t) * particles_size_;
		vku::CreateSSBO(physicalDevice, device, queue, commandPool,
			delta_x_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			deltaX,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			delta_x_ssbo_, delta_x_ssbo_memory_);

		// delta Y
		delta_y_ssbo_size_ = sizeof(uint32_t) * particles_size_;
		vku::CreateSSBO(physicalDevice, device, queue, commandPool,
			delta_y_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			deltaY,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			delta_y_ssbo_, delta_y_ssbo_memory_);

		// delta Z
		delta_z_ssbo_size_ = sizeof(uint32_t) * particles_size_;
		vku::CreateSSBO(physicalDevice, device, queue, commandPool,
			delta_z_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			deltaZ,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			delta_z_ssbo_, delta_z_ssbo_memory_);

		// delta count
		dcount_ssbo_size_ = sizeof(uint32_t) * particles_size_;
		vku::CreateSSBO(physicalDevice, device, queue, commandPool,
			dcount_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			dcount,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			dcount_ssbo_, dcount_ssbo_memory_);

		// edge
		edges_ssbo_size_ = sizeof(Edge) * edge.size();
		vku::CreateSSBO(physicalDevice, device, queue, commandPool,
			edges_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			edge,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			edges_ssbo_, edges_ssbo_memory_);
	}

	// Descriptor Sets
	{

		// Sim Params
		{
			vk::DescriptorSetAllocateInfo allocInfo{
				.descriptorPool = *descriptor_pool_,
				.descriptorSetCount = 1,
				.pSetLayouts = &*compute_.sim_params_set_layout
			};

			auto sets = vk::raii::DescriptorSets{ device, allocInfo };
			compute_.sim_params_set = std::move(sets.front());

			vk::DescriptorBufferInfo simParamsUboInfo{ *compute_.sim_params_ubo, 0, sizeof(Compute::SimParams) };
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = *compute_.sim_params_set,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
					.pBufferInfo = &simParamsUboInfo
				}
			};
			device.updateDescriptorSets(descriptorWrites, {});
		}

		// Cloth Compute
		{
			vk::DescriptorSetAllocateInfo allocInfo{
				.descriptorPool = *descriptor_pool_,
				.descriptorSetCount = 1,
				.pSetLayouts = &*compute_.cloth_compute_set_layout
			};

			auto sets = vk::raii::DescriptorSets{ device, allocInfo };
			compute_.cloth_compute_set = std::move(sets.front());

			vk::DescriptorBufferInfo positions(*positions_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo velocities(*velocities_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo inverseMass(*inverse_mass_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo deltaX(*delta_x_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo deltaY(*delta_y_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo deltaZ(*delta_z_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo dcount(*dcount_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo edge(*edges_ssbo_, 0, VK_WHOLE_SIZE);
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = *compute_.cloth_compute_set,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &positions
				},
				vk::WriteDescriptorSet{
					.dstSet = *compute_.cloth_compute_set,
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &velocities
				},
					vk::WriteDescriptorSet{
						.dstSet = *compute_.cloth_compute_set,
						.dstBinding = 2,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eStorageBuffer,
						.pBufferInfo = &inverseMass
				},
					vk::WriteDescriptorSet{
						.dstSet = *compute_.cloth_compute_set,
						.dstBinding = 3,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eStorageBuffer,
						.pBufferInfo = &deltaX
				},
					vk::WriteDescriptorSet{
						.dstSet = *compute_.cloth_compute_set,
						.dstBinding = 4,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eStorageBuffer,
						.pBufferInfo = &deltaY
				},
					vk::WriteDescriptorSet{
						.dstSet = *compute_.cloth_compute_set,
						.dstBinding = 5,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eStorageBuffer,
						.pBufferInfo = &deltaZ
				},
					vk::WriteDescriptorSet{
						.dstSet = *compute_.cloth_compute_set,
						.dstBinding = 6,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eStorageBuffer,
						.pBufferInfo = &dcount
				},
					vk::WriteDescriptorSet{
						.dstSet = *compute_.cloth_compute_set,
						.dstBinding = 7,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eStorageBuffer,
						.pBufferInfo = &edge
				},
			};
			device.updateDescriptorSets(descriptorWrites, {});
		}

		// Cloth Graphics
		{
			vk::DescriptorSetAllocateInfo allocInfo{
				.descriptorPool = *descriptor_pool_,
				.descriptorSetCount = 1,
				.pSetLayouts = &*graphics_.cloth_set_layout
			};

			auto sets = vk::raii::DescriptorSets{ device, allocInfo };
			graphics_.cloth_set = std::move(sets.front());

			vk::DescriptorImageInfo imageInfo{
				.sampler = *texture->texture_sampler_,
				.imageView = *texture->texture_image_view_,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			};
			vk::DescriptorBufferInfo positions(positions_ssbo_, 0, VK_WHOLE_SIZE);
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = *graphics_.cloth_set,
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eCombinedImageSampler,
					.pImageInfo = &imageInfo
				},
				vk::WriteDescriptorSet{
					.dstSet = *graphics_.cloth_set,
					.dstBinding = 2,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &positions
				}

			};
			device.updateDescriptorSets(descriptorWrites, {});
		}
	}

	// Compute Pipeline
	{
		// common pipeline layout
		{
			std::array<vk::DescriptorSetLayout, 2> setLayouts(*compute_.sim_params_set_layout, *compute_.cloth_compute_set_layout);

			vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 2, .pSetLayouts = setLayouts.data(), .pushConstantRangeCount = 0 };
			compute_.pipeline_layouts.common = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
		}

		// integrate
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device, vku::ReadFile("shaders/integrate.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.integrate = vk::raii::Pipeline(device, nullptr, pipelineInfo);
		}

		// clear lambdas
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device, vku::ReadFile("shaders/clear_lambdas.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.clear_lambdas = vk::raii::Pipeline(device, nullptr, pipelineInfo);
		}

		// clear deltas
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device, vku::ReadFile("shaders/clear_deltas.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.clear_deltas = vk::raii::Pipeline(device, nullptr, pipelineInfo);
		}

		// solve stretch xpbd
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device, vku::ReadFile("shaders/solve_stretch_xpbd.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.solve_stretch = vk::raii::Pipeline(device, nullptr, pipelineInfo);
		}

		// apply deltas
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device, vku::ReadFile("shaders/apply_deltas.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.apply_deltas = vk::raii::Pipeline(device, nullptr, pipelineInfo);
		}

		// collide sphere
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device, vku::ReadFile("shaders/collide_sphere.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.collide_sphere = vk::raii::Pipeline(device, nullptr, pipelineInfo);
		}

		// vel_update
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(device, vku::ReadFile("shaders/vel_update.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.vel_update = vk::raii::Pipeline(device, nullptr, pipelineInfo);
		}
	}

	// Graphics Pipeline
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

		// Cloth
		{
			// Shader
			auto vertCode = vku::ReadFile("shaders/cloth.vert.spv");
			auto fragCode = vku::ReadFile("shaders/cloth.frag.spv");

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
			std::array<vk::DescriptorSetLayout, 2> setLayouts(*globalSetLayout, *graphics_.cloth_set_layout);
			vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
				.setLayoutCount = 2,
				.pSetLayouts = setLayouts.data(),
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &pcRange
			};
			graphics_.pipeline_layouts.cloth = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

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
				.layout = graphics_.pipeline_layouts.cloth,
				.renderPass = nullptr },
			  {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchain->swapchain_surface_format_.format, .depthAttachmentFormat = depthFormat }
			};
			graphics_.pipelines.cloth = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		}
	}
}


void GpuSim::UpdateComputeUBO(uint32_t currentFrame)
{
	float frameDt = 1.0f / 240.0f;
	float substeps = 3.0f;

	compute_.sim_params.dt = frameDt / substeps;
	compute_.sim_params.inv_dt = 1.0f / compute_.sim_params.dt;
	compute_.sim_params.substeps = substeps;
	compute_.sim_params.iterations = 16.0f;

	compute_.sim_params.gravity = glm::vec4(0.0f, -9.8f, 0.0f, 0.0f);
	compute_.sim_params.num_particles = particles_size_;
	compute_.sim_params.num_edges = edges_size_;
	compute_.sim_params.damping = 0.05f;
	compute_.sim_params.collision_friction = 0.3f;

	const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * compute_.sim_params_slot_size);
	auto* dst = static_cast<std::byte*>(compute_.sim_params_ubo_mapped) + baseOffset;


	std::memcpy(dst, &compute_.sim_params, sizeof(Compute::SimParams));
}

void GpuSim::Record(uint32_t currentFrame)
{
	const auto& cmd = compute_.command_buffers[currentFrame];

	auto barrier_buffers = [&](std::span<const vk::BufferMemoryBarrier2> b) {
		vk::DependencyInfo dep{
			.bufferMemoryBarrierCount = static_cast<uint32_t>(b.size()),
			.pBufferMemoryBarriers = b.data(),
		};
		cmd.pipelineBarrier2(dep);
		};

	auto make_bmb = [](vk::Buffer buf, vk::DeviceSize size,
		vk::AccessFlags2 srcAcc, vk::PipelineStageFlags2 srcStage,
		vk::AccessFlags2 dstAcc, vk::PipelineStageFlags2 dstStage)
		{
			return vk::BufferMemoryBarrier2{
				.srcStageMask = srcStage,
				.srcAccessMask = srcAcc,
				.dstStageMask = dstStage,
				.dstAccessMask = dstAcc,
				.buffer = buf,
				.offset = 0,
				.size = size
			};
		};

	cmd.reset();
	cmd.begin({});

	uint32_t simparamOffset = currentFrame * static_cast<uint32_t>(compute_.sim_params_slot_size);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute_.pipeline_layouts.common, 0, { compute_.sim_params_set, compute_.cloth_compute_set }, { simparamOffset });

	auto ceil_div = [](uint32_t n, uint32_t d) { return (n + d - 1) / d; };
	uint32_t groupsP = ceil_div(particles_size_, 256u); // per-particle 커널용
	uint32_t groupsE = ceil_div(edges_size_, 256u); // per-edge 커널용

	// integrate
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_.pipelines.integrate);
	cmd.dispatch(groupsP, 1, 1);

	// xp : write -> read
	{
		std::array b = {
			make_bmb(*positions_ssbo_, positions_ssbo_size_,
					 vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eComputeShader,
					 vk::AccessFlagBits2::eShaderStorageRead,  vk::PipelineStageFlagBits2::eComputeShader)
		};
		barrier_buffers(b);
	}

	cmd.end();
}