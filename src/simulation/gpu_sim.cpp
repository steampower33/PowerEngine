#include "context.h"
#include "swapchain.h"
#include "texture_2d.h"
#include "texture_manager.h"
#include "model.h"
#include "model_manager.h"
#include "vulkan_utils.h"

#include "gpu_sim.h"

GpuSim::GpuSim(
	Context& context,
	Swapchain& swapchain,
	TextureManager& textureManager,
	vk::raii::DescriptorSetLayout& globalSetLayout,
	uint32_t Nx, uint32_t Ny, float spacing,
	std::vector<vk::Format>& formats)
{
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
			compute_.sim_params_set_layout = vk::raii::DescriptorSetLayout(context.device_, layoutInfo);
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
			compute_.cloth_compute_set_layout = vk::raii::DescriptorSetLayout(context.device_, layoutInfo);
		}

		// Cloth Rendering - Graphics
		{
			std::array<vk::DescriptorSetLayoutBinding, 2> layoutBindings{
				vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex },
				vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
			};
			counts_.sampler += 1;
			counts_.sb += 1;
			counts_.layout += 1;

			vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
			graphics_.cloth_set_layout = vk::raii::DescriptorSetLayout(context.device_, layoutInfo);
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

		descriptor_pool_ = vk::raii::DescriptorPool(context.device_, poolInfo);
	}

	// Sim Params UBO
	{
		compute_.sim_params_ubo.clear();
		compute_.sim_params_ubo_memory.clear();
		compute_.sim_params_ubo_mapped = nullptr;

		auto limits = context.physical_device_.getProperties().limits;
		compute_.sim_params_slot_size = (sizeof(Compute::SimParams) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = compute_.sim_params_slot_size * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context.physical_device_, context.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		compute_.sim_params_ubo = std::move(buffer);
		compute_.sim_params_ubo_memory = std::move(bufferMem);
		compute_.sim_params_ubo_mapped = compute_.sim_params_ubo_memory.mapMemory(0, totalSize);
	}

	// SSBOs
	{
		Nx_ = Nx;
		Ny_ = Ny;
		spacing_ = spacing;

		const int nxCells = Nx_;
		const int nyCells = Ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;            // 파티클 총수
		particles_size_ = N;

		// 입력
		positions_.resize(N);
		velocities_.resize(N);
		inverse_mass_.resize(N);

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * Nx_) * spacing_;   // (-Nx/2 .. +Nx/2) * spacing
				float py = 8.0f;
				float pz = (y - 0.5f * Ny_) * spacing_;   // (-Ny/2 .. +Ny/2) * spacing
				positions_[id] = { px, py, pz, 1.0f };
				velocities_[id] = glm::vec4(0);
				inverse_mass_[id] = 1.0f;
			}
		}

		// 인덱스 버퍼도 그대로
		std::vector<uint32_t> indices;
		indices.reserve(Nx_ * Ny_ * 6);
		for (int y = 0; y < Ny_; ++y) {
			for (int x = 0; x < Nx_; ++x) {
				uint32_t i0 = vid(x, y);
				uint32_t i1 = vid(x + 1, y);
				uint32_t i2 = vid(x, y + 1);
				uint32_t i3 = vid(x + 1, y + 1);
				indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
				indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
			}
		}
		indices_size_ = static_cast<uint32_t>(indices.size());

		vku::CreateIndexBuffer(context.physical_device_, context.device_, context.queue_, context.command_pool_, indices, index_buffer_, index_buffer_memory_);

		// 출력 SoA
		std::vector<glm::vec4> xPred(N, glm::vec4(0.0f));
		std::vector<float > deltaX(N, 0.0f), deltaY(N, 0.0f), deltaZ(N, 0.0f);
		std::vector<uint32_t>  dcount(N, 0);

		std::vector<std::pair<uint32_t, uint32_t>> pass[6];

		// (0) 세로 짝수
		for (int x = 0; x < nx1; ++x)
			for (int y = 0; y + 1 < ny1; y += 2)
				pass[0].push_back({ vid(x,y), vid(x,y + 1) });

		// (1) 세로 홀수
		for (int x = 0; x < nx1; ++x)
			for (int y = 1; y + 1 < ny1; y += 2)
				pass[1].push_back({ vid(x,y), vid(x,y + 1) });

		// (2) 가로 짝수
		for (int y = 0; y < ny1; ++y)
			for (int x = 0; x + 1 < nx1; x += 2)
				pass[2].push_back({ vid(x,y), vid(x + 1,y) });

		// (3) 가로 홀수
		for (int y = 0; y < ny1; ++y)
			for (int x = 1; x + 1 < nx1; x += 2)
				pass[3].push_back({ vid(x,y), vid(x + 1,y) });

		// (4) Diagonal
		for (int y = 0; y + 1 < ny1; ++y)
			for (int x = 0; x + 1 < nx1; ++x) {
				pass[4].push_back({ vid(x,y),     vid(x + 1,y + 1) }); // "\"
				pass[4].push_back({ vid(x + 1,y),   vid(x,  y + 1) }); // "/"
			}

		//// (5) 2-step bend
		//for (int x = 0; x < nx1; ++x)
		//	for (int y = 0; y + 2 < ny1; ++y)
		//		pass[5].push_back({ vid(x,y), vid(x,  y + 2) });
		//for (int y = 0; y < ny1; ++y)
		//	for (int x = 0; x + 2 < nx1; ++x)
		//		pass[5].push_back({ vid(x,y), vid(x + 2,y) });

		pass_offset_[0] = 0;

		for (int p = 0; p < 5; ++p) {
			for (auto [i, j] : pass[p]) {
				glm::vec3 pi = glm::vec3(positions_[i]);
				glm::vec3 pj = glm::vec3(positions_[j]);
				float rest = glm::length(pj - pi);
				edges_.push_back({ i, j, rest, 0.0f });
			}
			pass_offset_[p + 1] = static_cast<uint32_t>(edges_.size());
		}

		std::vector<Bend> bends;

		for (int y = 0; y < Ny_; ++y) {
			for (int x = 0; x < Nx_; ++x) {
				uint32_t i0 = vid(x, y);
				uint32_t i1 = vid(x + 1, y);
				uint32_t i2 = vid(x, y + 1);
				uint32_t i3 = vid(x + 1, y + 1);

				// 두 삼각형: (i0,i1,i2), (i1,i3,i2)
				uint32_t p1 = i1;   // hinge start
				uint32_t p2 = i2;   // hinge end
				uint32_t p3 = i0;   // opp of first tri
				uint32_t p4 = i3;   // opp of second tri

				// 평면 시작이면 0으로 충분. 필요하면 아래 함수로 실제 θ0 계산.
				float theta0 = 0.0f;
				//theta0 = ComputeRestAngle(p1,p2,p3,p4, positions_);

				bends.push_back({ p1,p2,p3,p4,theta0, 0.0f, glm::vec2(0.0f, 0.0f) });
			}
		}
		bends_size_ = static_cast<uint32_t>(bends.size());

		edge_size_ = static_cast<uint32_t>(edges_.size());

		sdfColliders.push_back(SDFCollider{ 0, glm::vec3(0.0f, 0.0f, 0.0f), 1.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), 0.0f });

		// position
		positions_ssbo_size_ = sizeof(glm::vec4) * N;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			positions_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			positions_,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			positions_ssbo_, positions_ssbo_memory_,
			&positions_staging_, &positions_staging_memory_);
		positions_staging_mapped_ = positions_staging_memory_.mapMemory(0, positions_ssbo_size_);

		// velocity
		velocities_ssbo_size_ = sizeof(glm::vec4) * N;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			velocities_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			velocities_,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			velocities_ssbo_, velocities_ssbo_memory_,
			&velocities_staging_, &velocities_staging_memory_);
		velocities_staging_mapped_ = velocities_staging_memory_.mapMemory(0, velocities_ssbo_size_);

		// inverse mass
		inverse_mass_ssbo_size_ = sizeof(float) * N;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			inverse_mass_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			inverse_mass_,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			inverse_mass_ssbo_, inverse_mass_ssbo_memory_,
			&inverse_mass_staging_, &inverse_mass_staging_memory_);
		inverse_mass_staging_mapped_ = inverse_mass_staging_memory_.mapMemory(0, inverse_mass_ssbo_size_);

		// delta X
		delta_x_ssbo_size_ = sizeof(float) * N;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			delta_x_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			deltaX,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			delta_x_ssbo_, delta_x_ssbo_memory_);

		// delta Y
		delta_y_ssbo_size_ = sizeof(float) * N;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			delta_y_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			deltaY,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			delta_y_ssbo_, delta_y_ssbo_memory_);

		// delta Z
		delta_z_ssbo_size_ = sizeof(float) * N;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			delta_z_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			deltaZ,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			delta_z_ssbo_, delta_z_ssbo_memory_);

		// delta count
		dcount_ssbo_size_ = sizeof(uint32_t) * N;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			dcount_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			dcount,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			dcount_ssbo_, dcount_ssbo_memory_);

		// edges
		edges_ssbo_size_ = sizeof(Edge) * edge_size_;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			edges_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			edges_,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			edges_ssbo_, edges_ssbo_memory_,
			&edges_staging_, &edges_staging_memory_);
		edges_staging_mapped_ = edges_staging_memory_.mapMemory(0, edges_ssbo_size_);

		// Pred Position
		pred_positions_ssbo_size_ = sizeof(glm::vec4) * N;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			pred_positions_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			xPred,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			pred_positions_ssbo_, pred_positions_ssbo_memory_);

		// Bend
		bends_ssbo_size_ = sizeof(Bend) * bends_size_;
		vku::CreateSSBO(context.physical_device_, context.device_, context.queue_, context.command_pool_,
			bends_ssbo_size_,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			bends,
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			bends_ssbo_, bends_ssbo_memory_);
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

			auto sets = vk::raii::DescriptorSets{ context.device_, allocInfo };
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
			context.device_.updateDescriptorSets(descriptorWrites, {});
		}

		// Cloth Compute
		{
			vk::DescriptorSetAllocateInfo allocInfo{
				.descriptorPool = *descriptor_pool_,
				.descriptorSetCount = 1,
				.pSetLayouts = &*compute_.cloth_compute_set_layout
			};

			auto sets = vk::raii::DescriptorSets{ context.device_, allocInfo };
			compute_.cloth_compute_set = std::move(sets.front());

			vk::DescriptorBufferInfo positions(*positions_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo velocities(*velocities_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo inverseMass(*inverse_mass_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo deltaX(*delta_x_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo deltaY(*delta_y_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo deltaZ(*delta_z_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo dcount(*dcount_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo edge(*edges_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo prevPositions(*pred_positions_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorBufferInfo bend(*bends_ssbo_, 0, VK_WHOLE_SIZE);
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
					vk::WriteDescriptorSet{
						.dstSet = *compute_.cloth_compute_set,
						.dstBinding = 8,
						.dstArrayElement = 0,
						.descriptorCount = 1,
						.descriptorType = vk::DescriptorType::eStorageBuffer,
						.pBufferInfo = &prevPositions
				},
					vk::WriteDescriptorSet{
						.dstSet = *compute_.cloth_compute_set,
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
				.pSetLayouts = &*graphics_.cloth_set_layout
			};

			auto sets = vk::raii::DescriptorSets{ context.device_, allocInfo };
			graphics_.cloth_set = std::move(sets.front());

			vk::DescriptorBufferInfo positions(positions_ssbo_, 0, VK_WHOLE_SIZE);
			vk::DescriptorImageInfo imageInfo{
					.sampler = *textureManager.textures_[0]->texture_sampler_,
					.imageView = *textureManager.textures_[0]->texture_image_view_,
					.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			};
			std::array descriptorWrites{
				vk::WriteDescriptorSet{
					.dstSet = *graphics_.cloth_set,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &positions
				},
				vk::WriteDescriptorSet{
					.dstSet = *graphics_.cloth_set,
					.dstBinding = 1,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eCombinedImageSampler,
					.pImageInfo = &imageInfo,
				},

			};
			context.device_.updateDescriptorSets(descriptorWrites, {});
		}
	}

	// Compute Pipeline
	{
		// common pipeline layout
		{
			std::array<vk::DescriptorSetLayout, 2> setLayouts(*compute_.sim_params_set_layout, *compute_.cloth_compute_set_layout);

			vk::PushConstantRange pcRange{
				.stageFlags = vk::ShaderStageFlagBits::eCompute,
				.offset = 0,
				.size = static_cast<uint32_t>(sizeof(Compute::PC))
			};

			vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
				.setLayoutCount = 2,
				.pSetLayouts = setLayouts.data(),
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &pcRange
			};
			compute_.pipeline_layouts.common = vk::raii::PipelineLayout(context.device_, pipelineLayoutInfo);
		}

		// clear_lambdas
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/clear_lambdas.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.clear_lambdas = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
		}

		// integrate
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/integrate.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.integrate = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
		}

		// solve_coloring
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_coloring.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common,
			};
			compute_.pipelines.solve_coloring = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
		}

		// solve_atomic
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_atomic.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common,
			};
			compute_.pipelines.solve_atomic = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
		}

		// solve_bend
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/solve_bend.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common,
			};
			compute_.pipelines.solve_bend = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
		}

		// apply deltas
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/apply_deltas.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.apply_deltas = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
		}
		
		// collide_sdf
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/collide_sdf.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.collide_sdf = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
		}

		// update velocity
		{
			vk::raii::ShaderModule shaderModule = vku::CreateShaderModule(context.device_, vku::ReadFile("shaders/spv/update_velocity.comp.spv"));

			vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "main" };

			vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *compute_.pipeline_layouts.common };
			compute_.pipelines.update_velocity = vk::raii::Pipeline(context.device_, nullptr, pipelineInfo);
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
			a.blendEnable = vk::False;  // G-buffer라 blending 불필요
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

			// push constant 범위: VS에서만 사용(필요하면 FS도 추가)
			vk::PushConstantRange pcRange{
				.stageFlags = vk::ShaderStageFlagBits::eVertex,
				.offset = 0,
				.size = static_cast<uint32_t>(sizeof(ClothPC))
			};
			cloth_pc_.nx1 = Nx_ + 1;
			cloth_pc_.ny1 = Ny_ + 1;

			// Pipeline Layout
			std::array<vk::DescriptorSetLayout, 2> setLayouts(*globalSetLayout, *graphics_.cloth_set_layout);
			vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
				.setLayoutCount = 2,
				.pSetLayouts = setLayouts.data(),
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &pcRange
			};
			graphics_.pipeline_layouts.cloth_solid = vk::raii::PipelineLayout(context.device_, pipelineLayoutInfo);

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
				.layout = graphics_.pipeline_layouts.cloth_solid,
				.renderPass = nullptr },
			  {.colorAttachmentCount = static_cast<uint32_t>(formats.size()), .pColorAttachmentFormats = formats.data(), .depthAttachmentFormat = depthFormat}
			};
			graphics_.pipelines.cloth_solid = vk::raii::Pipeline(context.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

			rasterizer.polygonMode = vk::PolygonMode::eLine,

			graphics_.pipelines.cloth_wireframe = vk::raii::Pipeline(context.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

			rasterizer.polygonMode = vk::PolygonMode::ePoint,
			graphics_.pipelines.cloth_point = vk::raii::Pipeline(context.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
		}
	}
}

void GpuSim::UpdateComputeUBO(uint32_t currentFrame, std::unique_ptr<Model>& model)
{
	compute_.sim_params.dt = 1 / deviding_dt_;
	compute_.sim_params.numParticles = particles_size_;
	compute_.sim_params.numEdges = edge_size_;
	compute_.sim_params.numBends = bends_size_;
	compute_.sim_params.maxSpeed = 2 * spacing_ * 1.5f / compute_.sim_params.dt;
	compute_.sim_params.sphereCenter = glm::vec4(model->position_, 0.0f);
	compute_.sim_params.sphereRadius = model->radius_;

	const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * compute_.sim_params_slot_size);
	auto* dst = static_cast<std::byte*>(compute_.sim_params_ubo_mapped) + baseOffset;

	std::memcpy(dst, &compute_.sim_params, sizeof(Compute::SimParams));
}

void GpuSim::GraphicsRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::DescriptorSet& globalSet, uint32_t globalOffset)
{
	// Cloth
	{
		if (is_wireframe_)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_.pipelines.cloth_wireframe);
		else if (is_point_)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_.pipelines.cloth_point);
		else
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_.pipelines.cloth_solid);

		// Global Set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			graphics_.pipeline_layouts.cloth_solid,
			0,
			{ *globalSet },
			{ globalOffset }
		);
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			graphics_.pipeline_layouts.cloth_solid,
			1,
			{ *graphics_.cloth_set },
			{ }
		);

		cmd.pushConstants<ClothPC>(
			*graphics_.pipeline_layouts.cloth_solid,
			vk::ShaderStageFlagBits::eVertex,
			/*offset=*/0,
			cloth_pc_
		);

		cmd.bindIndexBuffer(*index_buffer_, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(indices_size_, 1, 0, 0, 0);
	}
}

void GpuSim::UpdateTestScene(const vk::raii::CommandBuffer& cmd, vku::TestScene& testScene)
{
	if (testScene.sphereCollision)
	{
		testScene.sphereCollision = false;

		const int nxCells = Nx_;
		const int nyCells = Ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;            // 파티클 총수

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * Nx_) * spacing_;   // (-Nx/2 .. +Nx/2) * spacing
				float py = 8.0f;
				float pz = (y - 0.5f * Ny_) * spacing_;   // (-Ny/2 .. +Ny/2) * spacing
				positions_[id] = { px, py, pz, 1.0f };
				velocities_[id] = glm::vec4(0);
				inverse_mass_[id] = 1.0f;
			}
		}

		// 인덱스 버퍼도 그대로
		std::vector<uint32_t> indices;
		indices.reserve(Nx_ * Ny_ * 6);
		for (int y = 0; y < Ny_; ++y) {
			for (int x = 0; x < Nx_; ++x) {
				uint32_t i0 = vid(x, y);
				uint32_t i1 = vid(x + 1, y);
				uint32_t i2 = vid(x, y + 1);
				uint32_t i3 = vid(x + 1, y + 1);
				indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
				indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
			}
		}

		for (int i = 0; i < edge_size_; i++)
		{
			edges_[i].lambda = 0.0f;
		}

		vku::CopyStagingToSSBO(cmd, positions_ssbo_size_, positions_staging_mapped_, positions_, positions_staging_, positions_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderStorageRead);

		vku::CopyStagingToSSBO(cmd, velocities_ssbo_size_, velocities_staging_mapped_, velocities_, velocities_staging_, velocities_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

		vku::CopyStagingToSSBO(cmd, inverse_mass_ssbo_size_, inverse_mass_staging_mapped_, inverse_mass_, inverse_mass_staging_, inverse_mass_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

		vku::CopyStagingToSSBO(cmd, edges_ssbo_size_, edges_staging_mapped_, edges_, edges_staging_, edges_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	}
	else if (testScene.pinnedCorner)
	{
		testScene.pinnedCorner = false;

		const int nxCells = Nx_;
		const int nyCells = Ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;            // 파티클 총수

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * Nx_) * spacing_;   // (-Nx/2 .. +Nx/2) * spacing
				float py = 8.0f;
				float pz = (y - 0.5f * Ny_) * spacing_;   // (-Ny/2 .. +Ny/2) * spacing
				positions_[id] = { px, py, pz, 1.0f };
				velocities_[id] = glm::vec4(0);
				inverse_mass_[id] = 1.0f;
			}
		}

		// 인덱스 버퍼도 그대로
		std::vector<uint32_t> indices;
		indices.reserve(Nx_ * Ny_ * 6);
		for (int y = 0; y < Ny_; ++y) {
			for (int x = 0; x < Nx_; ++x) {
				uint32_t i0 = vid(x, y);
				uint32_t i1 = vid(x + 1, y);
				uint32_t i2 = vid(x, y + 1);
				uint32_t i3 = vid(x + 1, y + 1);
				indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
				indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
			}
		}

		inverse_mass_[0] = 0.0f;
		inverse_mass_[nx1 - 1] = 0.0f;
		inverse_mass_[(ny1 - 1) * nx1] = 0.0f;
		inverse_mass_[(ny1 - 1) * nx1 + nx1 - 1] = 0.0f;

		for (int i = 0; i < edge_size_; i++)
		{
			edges_[i].lambda = 0.0f;
		}

		vku::CopyStagingToSSBO(cmd, positions_ssbo_size_, positions_staging_mapped_, positions_, positions_staging_, positions_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderStorageRead);

		vku::CopyStagingToSSBO(cmd, velocities_ssbo_size_, velocities_staging_mapped_, velocities_, velocities_staging_, velocities_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

		vku::CopyStagingToSSBO(cmd, inverse_mass_ssbo_size_, inverse_mass_staging_mapped_, inverse_mass_, inverse_mass_staging_, inverse_mass_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

		vku::CopyStagingToSSBO(cmd, edges_ssbo_size_, edges_staging_mapped_, edges_, edges_staging_, edges_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);
	}
	else if (testScene.topPinnedCorner)
	{
		testScene.topPinnedCorner = false;

		const int nxCells = Nx_;
		const int nyCells = Ny_;
		const int nx1 = nxCells + 1;
		const int ny1 = nyCells + 1;

		auto vid = [&](int x, int y) { return uint32_t(y * nx1 + x); };

		const uint32_t N = nx1 * ny1;            // 파티클 총수

		for (int y = 0; y < ny1; ++y) {
			for (int x = 0; x < nx1; ++x) {
				uint32_t id = vid(x, y);
				float px = (x - 0.5f * Nx_) * spacing_;   // (-Nx/2 .. +Nx/2) * spacing
				float py = 8.0f;
				float pz = (y - 0.5f * Ny_) * spacing_;   // (-Ny/2 .. +Ny/2) * spacing
				positions_[id] = { px, py, pz, 1.0f };
				velocities_[id] = glm::vec4(0);
				inverse_mass_[id] = 1.0f;
			}
		}

		// 인덱스 버퍼도 그대로
		std::vector<uint32_t> indices;
		indices.reserve(Nx_ * Ny_ * 6);
		for (int y = 0; y < Ny_; ++y) {
			for (int x = 0; x < Nx_; ++x) {
				uint32_t i0 = vid(x, y);
				uint32_t i1 = vid(x + 1, y);
				uint32_t i2 = vid(x, y + 1);
				uint32_t i3 = vid(x + 1, y + 1);
				indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
				indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
			}
		}

		//inverse_mass_[0] = 0.0f;
		//inverse_mass_[nx1 - 1] = 0.0f;
		inverse_mass_[(ny1 - 1) * nx1] = 0.0f;
		inverse_mass_[(ny1 - 1) * nx1 + nx1 - 1] = 0.0f;

		for (int i = 0; i < edge_size_; i++)
		{
			edges_[i].lambda = 0.0f;
		}

		vku::CopyStagingToSSBO(cmd, positions_ssbo_size_, positions_staging_mapped_, positions_, positions_staging_, positions_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderStorageRead);

		vku::CopyStagingToSSBO(cmd, velocities_ssbo_size_, velocities_staging_mapped_, velocities_, velocities_staging_, velocities_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

		vku::CopyStagingToSSBO(cmd, inverse_mass_ssbo_size_, inverse_mass_staging_mapped_, inverse_mass_, inverse_mass_staging_, inverse_mass_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

		vku::CopyStagingToSSBO(cmd, edges_ssbo_size_, edges_staging_mapped_, edges_, edges_staging_, edges_ssbo_,
			vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);
	}
}

void GpuSim::ComputeRecord(uint32_t currentFrame, const vk::raii::CommandBuffer& cmd, vk::raii::QueryPool& timestampPool, uint32_t& timestampSteps, vku::TestScene& testScene)
{
	cmd.reset();
	cmd.begin({});

	UpdateTestScene(cmd, testScene);

	timestampSteps = 0;
	uint32_t kSlotsPerIterPair = 4 + iter_contraint_count_ * iterations_ + 2;
	const auto stage = vk::PipelineStageFlagBits2::eComputeShader;
	auto TS = [&](uint32_t& idx) {
		cmd.writeTimestamp2(stage, *timestampPool, idx++);
		};

	cmd.resetQueryPool(*timestampPool, 0, kSlotsPerIterPair);

	uint32_t simparamOffset = currentFrame * static_cast<uint32_t>(compute_.sim_params_slot_size);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, compute_.pipeline_layouts.common, 0, { compute_.sim_params_set, compute_.cloth_compute_set }, { simparamOffset });

	auto ceil_div = [](uint32_t n, uint32_t d) { return (n + d - 1) / d; };
	uint32_t groupsP = ceil_div(particles_size_, 256u);
	uint32_t groupsEdges = ceil_div(edge_size_, 256u);

	// 1. Integrate
	// x : write -> read
	TS(timestampSteps);
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_.pipelines.integrate);
	cmd.dispatch(groupsP, 1, 1);
	TS(timestampSteps);
	vku::barrier2(cmd,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	// 2. Clear Lambdas
	TS(timestampSteps);
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_.pipelines.clear_lambdas);
	cmd.dispatch(groupsEdges, 1, 1);
	TS(timestampSteps);
	vku::barrier2(cmd,  // write->read
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);

	for (uint32_t iter = 0; iter < iterations_; iter++)
	{
		// 3. Solve Coloring - Stretch
		TS(timestampSteps);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_.pipelines.solve_coloring);

		for (uint32_t c = 0; c < 4; ++c) {
			uint32_t base = pass_offset_[c];
			uint32_t count = pass_offset_[c + 1] - pass_offset_[c];
			if (!count) continue;

			compute_.pc_.base = base;
			compute_.pc_.count = count;	
			compute_.pc_.compliance = 1e-10f;

			cmd.pushConstants<Compute::PC>(*compute_.pipeline_layouts.common, vk::ShaderStageFlagBits::eCompute, 0u, compute_.pc_);

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
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_.pipelines.solve_atomic);
			uint32_t base = pass_offset_[4];
			uint32_t count = pass_offset_[5] - pass_offset_[4];
			compute_.pc_.base = base;
			compute_.pc_.count = count;
			compute_.pc_.compliance = 5e-9f;
			cmd.pushConstants<Compute::PC>(*compute_.pipeline_layouts.common,
				vk::ShaderStageFlagBits::eCompute, 0u, compute_.pc_);

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
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_.pipelines.solve_bend);
			uint32_t base = 0;
			uint32_t count = bends_size_;
			compute_.pc_.base = base;
			compute_.pc_.count = count;
			compute_.pc_.compliance = bendCompliance;
			cmd.pushConstants<Compute::PC>(*compute_.pipeline_layouts.common,
				vk::ShaderStageFlagBits::eCompute, 0u, compute_.pc_);

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
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_.pipelines.apply_deltas);
		cmd.dispatch(groupsP, 1, 1);
		TS(timestampSteps);
		vku::barrier2(cmd,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageWrite,
			vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

		// 7. Collide SDF
		TS(timestampSteps);
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_.pipelines.collide_sdf);
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
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_.pipelines.update_velocity);
	cmd.dispatch(groupsP, 1, 1);
	TS(timestampSteps);
	vku::barrier2(cmd,
		vk::PipelineStageFlagBits2::eComputeShader,
		vk::AccessFlagBits2::eShaderStorageWrite,
		vk::PipelineStageFlagBits2::eVertexShader,
		vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

	cmd.end();
}
