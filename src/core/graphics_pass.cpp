#include "context.h"
#include "camera.h"
#include "swapchain.h"
#include "vertex.h"
#include "texture.h"
#include "texture_manager.h"
#include "model.h"
#include "model_manager.h"
#include "particle_manager.h"
#include "model_loader.h"

#include "graphics_pass.h"

GraphicsPass::GraphicsPass(Context& context, Swapchain& swapchain, TextureManager& textureManager, ModelManager& modelManager, ParticleManager& particleManager)
	: context_(context), swapchain_(swapchain), texture_manager_(textureManager), model_manager_(modelManager), particle_manager_(particleManager)
{
	//msaa_samples_ = vku::GetMaxUsableSampleCount(context_.physical_device_.getProperties());

	CreateCommandBuffers();
	CreateQueryPool();
	CreateDescriptorSetLayout();
	CreateDescriptorPools();
	CreateUniformBuffers();
	CreateGeometryBuffers();
	CreateDepthResources();
	CreateShadowResources();
	CreateDescriptorSets();
	CreateGraphicsPipelines();
}

GraphicsPass::~GraphicsPass()
{

}

// Very Naive Method
void GraphicsPass::UpdateGraphicsUBO(uint32_t currentFrame, Camera& camera, bool paused)
{
	// Global UBO
	{
		const uint32_t globalOffset = static_cast<uint32_t>(currentFrame * ubo_size_.global);
		auto* dst = static_cast<std::byte*>(ubo_mapped_.global) + globalOffset;

		ubo_datas_.global.view = camera.View();
		ubo_datas_.global.proj = camera.Proj(swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height);

		std::memcpy(dst, &ubo_datas_.global, sizeof(ubo_data::Global));
	}

	// Light
	{
		ubo_datas_.light.camera_pos = glm::vec4(camera.position, 0.0f);
		ubo_datas_.light.inv_view_proj = glm::inverse(ubo_datas_.global.proj * ubo_datas_.global.view);

		glm::vec3 dir = glm::normalize(ubo_datas_.light.direction);
		glm::vec3 up = (std::abs(dir.y) > 0.99f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
		glm::mat4 lightView = glm::lookAt(ubo_datas_.light.position,
			ubo_datas_.light.position + dir,
			up);

		glm::mat4 lightProj = glm::perspectiveRH_ZO(glm::radians(90.0f), 1.0f, 0.1f, 1000.0f);
		lightProj[1][1] *= -1.0f;

		ubo_datas_.light.light_view_proj = lightProj * lightView;
		ubo_datas_.light.shadow_map_inv_size = glm::vec2(1.0f / float(shadow_extent_.width), 1.0f / float(shadow_extent_.height));

		const uint32_t lightOffset = static_cast<uint32_t>(currentFrame * ubo_size_.light);
		auto* dst = static_cast<std::byte*>(ubo_mapped_.light) + lightOffset;
		std::memcpy(dst, &ubo_datas_.light, sizeof(ubo_data::Light));
	}

	// Skybox
	{
		const uint32_t skyboxOffset = static_cast<uint32_t>(currentFrame * ubo_size_.skybox);
		auto* dst = static_cast<std::byte*>(ubo_mapped_.skybox) + skyboxOffset;

		auto& tm = texture_manager_;

		if (tm.skybox_enable_.morning)
		{
			tm.skybox_enable_.morning = false;
			ubo_datas_.skybox.env_idx = tm.skybox_index_.morning_env;
			ubo_datas_.skybox.specular_idx = tm.skybox_index_.morning_specular;
			ubo_datas_.skybox.diffuse_idx = tm.skybox_index_.morning_diffuse;
			ubo_datas_.skybox.specular_mip_levels = tm.tex_env_[tm.skybox_index_.morning_specular]->mip_levels_;
		}

		if (tm.skybox_enable_.evening)
		{
			tm.skybox_enable_.evening = false;
			ubo_datas_.skybox.env_idx = tm.skybox_index_.evening_env;
			ubo_datas_.skybox.specular_idx = tm.skybox_index_.evening_specular;
			ubo_datas_.skybox.diffuse_idx = tm.skybox_index_.evening_diffuse;
			ubo_datas_.skybox.specular_mip_levels = tm.tex_env_[tm.skybox_index_.evening_specular]->mip_levels_;
		}

		if (tm.skybox_enable_.night)
		{
			tm.skybox_enable_.night = false;
			ubo_datas_.skybox.env_idx = tm.skybox_index_.night_env;
			ubo_datas_.skybox.specular_idx = tm.skybox_index_.night_specular;
			ubo_datas_.skybox.diffuse_idx = tm.skybox_index_.night_diffuse;
			ubo_datas_.skybox.specular_mip_levels = tm.tex_env_[tm.skybox_index_.night_specular]->mip_levels_;
		}

		std::memcpy(dst, &ubo_datas_.skybox, sizeof(ubo_data::SkyBox));
	}

	{
		auto& pm = particle_manager_;

		const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * pm.clothes_.size() * ubo_size_.cloth);
		auto* base = static_cast<std::byte*>(ubo_mapped_.cloth) + baseOffset;

		for (uint32_t i = 0; i < pm.clothes_.size(); i++)
		{
			auto* dst = base + ubo_size_.cloth * i;
			std::memcpy(dst, &pm.clothes_[i].ubo_data, ubo_size_.cloth);
		}
	}

	// Model UBO
	{
		const uint32_t baseModelOffset = static_cast<uint32_t>(currentFrame * ubo_size_.model * model_manager_.kMaxModels);
		for (uint32_t i = 0; i < model_manager_.models_.size(); i++)
		{
			auto& model = *model_manager_.models_[i];

			if (!model.render_) continue;

			const uint32_t modelOff = baseModelOffset + i * static_cast<uint32_t>(ubo_size_.model);
			auto* dst = static_cast<std::byte*>(ubo_mapped_.model) + modelOff;

			model.ubo_data.world = model.world_;

			std::memcpy(dst, &model.ubo_data, ubo_size_.model);

			if (model.model_type_ == ModelType::SKINNED)
			{
				if (model.do_animation)
				{
					model.current_time_ += 1.0f / 240.0f;
					model.ApplyAnimation(0, model.current_time_);
					model.capsule_collision_update_ = true;
				}

				if (model.capsule_collision_collide_)
				{
					model.UpdateCapsuleCollidersFromBones();
				}

				auto& jm = model.model_loader_->skin_[0].jointMatrices;

				const uint32_t skinnedModelOff =
					static_cast<uint32_t>(currentFrame * ubo_size_.skinned_model);
				auto* dst = static_cast<std::byte*>(ubo_mapped_.skinned_model) + skinnedModelOff;

				size_t boneCount = jm.size();

				size_t copySize = boneCount * sizeof(glm::mat4);

				std::memcpy(dst, jm.data(), copySize);
			}
		}
	}

	// grid
	{
		const uint32_t globalOffset = static_cast<uint32_t>(currentFrame * ubo_size_.grid);
		auto* dst = static_cast<std::byte*>(ubo_mapped_.grid) + globalOffset;

		ubo_datas_.grid.view = camera.View();
		ubo_datas_.grid.proj = camera.Proj(swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height);
		ubo_datas_.grid.camera_pos = camera.position;

		std::memcpy(dst, &ubo_datas_.grid, sizeof(ubo_data::Grid));
	}
}

// ============================
// ========== Record ==========
// ============================
void GraphicsPass::RecordGraphicsCommandBuffer(uint32_t imageIndex, uint32_t currentFrame)
{
	const auto& cmd = cmds_[currentFrame];
	timestamp_steps_ = 0;
	uint32_t slots = 2;
	const auto stage = vk::PipelineStageFlagBits2::eComputeShader;
	auto TS = [&](uint32_t& idx) {
		cmd.writeTimestamp2(stage, *timestamp_pool_, idx++);
		};

	cmd.reset();
	cmd.begin({});

	cmd.resetQueryPool(*timestamp_pool_, 0, slots);

	TS(timestamp_steps_);

	ShadowDepthOnlyPass(cmd, currentFrame);
	PreMainRenderPass(cmd, currentFrame);
	MainRenderPass(cmd, currentFrame);
	PostMainRenderPass(cmd, imageIndex);
	LightingPass(cmd, imageIndex, currentFrame);
	InfiniteGridPass(cmd, imageIndex, currentFrame);

	{

		vku::TransitionImageLayout(
			swapchain_.swapchain_images_[imageIndex],
			cmd,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			{},
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eBottomOfPipe
		);

		TS(timestamp_steps_);

		cmd.end();

		frame_counter_++;
	}
}

void GraphicsPass::CalculateGpuTime()
{
	float nsPerTick = context_.physical_device_.getProperties().limits.timestampPeriod;
	float toMs = nsPerTick / 1e6f;

	if (timestamp_steps_ <= 0) return;

	uint32_t numTimestamp = timestamp_steps_;
	std::vector<uint64_t> ts(numTimestamp);

	VkResult res = vkGetQueryPoolResults(
		static_cast<VkDevice>(*context_.device_),
		static_cast<VkQueryPool>(*timestamp_pool_),
		0, numTimestamp,
		ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
	);

	auto delta_ms = [&](uint32_t i0, uint32_t i1) {
		return (ts[i1] - ts[i0]) * toMs;
		};

	pass_total_time_ = delta_ms(0, 1);
}

// ============================
// ========== Create ==========
// ============================
void GraphicsPass::CreateCommandBuffers()
{
	cmds_.clear();
	vk::CommandBufferAllocateInfo allocInfo{};
	allocInfo.commandPool = *context_.command_pool_;
	allocInfo.level = vk::CommandBufferLevel::ePrimary;
	allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
	cmds_ = vk::raii::CommandBuffers(context_.device_, allocInfo);
}

void GraphicsPass::CreateQueryPool() {

	vk::QueryPoolCreateInfo queryInfo = {};
	queryInfo.queryType = vk::QueryType::eTimestamp;
	queryInfo.queryCount = 2;

	timestamp_pool_ = context_.device_.createQueryPool(queryInfo);
}

void GraphicsPass::CreateDescriptorSetLayout()
{
	// Global UBO - Graphics
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.global = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Model UBO
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.model = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Light UBO + G-buffers + Shadow Depth Only Buffer
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				1,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				2,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				3,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				4,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				5,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
			vk::DescriptorSetLayoutBinding(
				6,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				vk::ShaderStageFlagBits::eFragment,
				nullptr
			),
		};

		counts_.ubo += 1;
		counts_.sampler += 6;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.lighting =
			vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Skybox UBO
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.skybox = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Cloth
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
			vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex },
			vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex }
		};
		counts_.ubo_dynamic += particle_manager_.clothes_.size() * MAX_FRAMES_IN_FLIGHT;
		counts_.sb += 2;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.cloth = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Softbody
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
			vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex },
			vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex },
		};
		counts_.ubo_dynamic += particle_manager_.softbodies_.size() * MAX_FRAMES_IN_FLIGHT;
		counts_.sb += 2;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.softbody = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Skinned Model
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eVertex,
				nullptr
			)
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{
			.bindingCount = static_cast<uint32_t>(layoutBindings.size()),
			.pBindings = layoutBindings.data()
		};
		set_layouts_.skinned_model = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// Shadow
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex },
		};
		counts_.sb += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.shadow_particle = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}

	// grid
	{
		std::array layoutBindings{
			vk::DescriptorSetLayoutBinding(
				0,
				vk::DescriptorType::eUniformBufferDynamic,
				1,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				nullptr
			)
		};
		counts_.ubo_dynamic += 1;
		counts_.layout += 1;

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
		set_layouts_.grid = vk::raii::DescriptorSetLayout(context_.device_, layoutInfo);
	}
}

void GraphicsPass::CreateDescriptorPools() {

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

void GraphicsPass::CreateUniformBuffers()
{
	// Global
	{
		ubos_.global.clear();
		ubo_memories_.global.clear();
		ubo_mapped_.global = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.global = (sizeof(ubo_data::Global) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.global * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.global = std::move(buffer);
		ubo_memories_.global = std::move(bufferMem);
		ubo_mapped_.global = ubo_memories_.global.mapMemory(0, totalSize);
	}

	// Model
	{
		ubos_.model.clear();
		ubo_memories_.model.clear();
		ubo_mapped_.model = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.model = (sizeof(ubo_data::Model) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.model * MAX_FRAMES_IN_FLIGHT * model_manager_.kMaxModels;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.model = std::move(buffer);
		ubo_memories_.model = std::move(bufferMem);
		ubo_mapped_.model = ubo_memories_.model.mapMemory(0, totalSize);
	}

	// Light
	{
		ubos_.light.clear();
		ubo_memories_.light.clear();
		ubo_mapped_.light = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.light = (sizeof(ubo_data::Light) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.light * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.light = std::move(buffer);
		ubo_memories_.light = std::move(bufferMem);
		ubo_mapped_.light = ubo_memories_.light.mapMemory(0, totalSize);

		ubo_datas_.light.ggx_brdf_idx = texture_manager_.brdf_index_.ggx;
		ubo_datas_.light.charlie_brdf_idx = texture_manager_.brdf_index_.charlie;
		ubo_datas_.light.sheen_e_brdf_idx = texture_manager_.brdf_index_.sheen_e;
	}

	// Skybox
	{
		ubos_.skybox.clear();
		ubo_memories_.skybox.clear();
		ubo_mapped_.skybox = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.skybox = (sizeof(ubo_data::SkyBox) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.skybox * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.skybox = std::move(buffer);
		ubo_memories_.skybox = std::move(bufferMem);
		ubo_mapped_.skybox = ubo_memories_.skybox.mapMemory(0, totalSize);

		ubo_datas_.skybox.env_idx = texture_manager_.skybox_index_.morning_env;
		ubo_datas_.skybox.specular_idx = texture_manager_.skybox_index_.morning_specular;
		ubo_datas_.skybox.diffuse_idx = texture_manager_.skybox_index_.morning_diffuse;
		ubo_datas_.skybox.specular_mip_levels = texture_manager_.tex2d_[texture_manager_.skybox_index_.morning_specular]->mip_levels_;

		auto* dst = static_cast<std::byte*>(ubo_mapped_.skybox);
		std::memcpy(dst, &ubo_datas_.skybox, ubo_size_.skybox);

		dst += ubo_size_.skybox;
		std::memcpy(dst, &ubo_datas_.skybox, ubo_size_.skybox);
	}

	// Cloth
	{
		ubos_.cloth.clear();
		ubo_memories_.cloth.clear();
		ubo_mapped_.cloth = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.cloth = (sizeof(ubo_data::Model) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = particle_manager_.clothes_.size() * ubo_size_.cloth * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.cloth = std::move(buffer);
		ubo_memories_.cloth = std::move(bufferMem);
		ubo_mapped_.cloth = ubo_memories_.cloth.mapMemory(0, totalSize);
	}

	// Softbody
	{
		ubos_.softbody.clear();
		ubo_memories_.softbody.clear();
		ubo_mapped_.softbody = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.softbody = (sizeof(ubo_data::Model) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = particle_manager_.softbodies_.size() * ubo_size_.softbody * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.softbody = std::move(buffer);
		ubo_memories_.softbody = std::move(bufferMem);
		ubo_mapped_.softbody = ubo_memories_.softbody.mapMemory(0, totalSize);
	}

	// Skinned Model
	{
		ubos_.skinned_model.clear();
		ubo_memories_.skinned_model.clear();
		ubo_mapped_.skinned_model = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		auto kMaxJoints = 128;
		ubo_size_.skinned_model = (sizeof(glm::mat4) * kMaxJoints + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.skinned_model * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.skinned_model = std::move(buffer);
		ubo_memories_.skinned_model = std::move(bufferMem);
		ubo_mapped_.skinned_model = ubo_memories_.skinned_model.mapMemory(0, totalSize);
	}

	// grid
	{
		ubos_.grid.clear();
		ubo_memories_.grid.clear();
		ubo_mapped_.grid = nullptr;

		auto limits = context_.physical_device_.getProperties().limits;
		ubo_size_.grid = (sizeof(ubo_data::Grid) + limits.minUniformBufferOffsetAlignment - 1)
			& ~(limits.minUniformBufferOffsetAlignment - 1);
		vk::DeviceSize totalSize = ubo_size_.grid * MAX_FRAMES_IN_FLIGHT;

		vk::raii::Buffer buffer({});
		vk::raii::DeviceMemory bufferMem({});
		vku::CreateBuffer(context_.physical_device_, context_.device_, totalSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
		ubos_.grid = std::move(buffer);
		ubo_memories_.grid = std::move(bufferMem);
		ubo_mapped_.grid = ubo_memories_.grid.mapMemory(0, totalSize);
	}
}

void GraphicsPass::CreateDescriptorSets()
{
	// Global UBO
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.global
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.global = std::move(sets.front());

		vk::DescriptorBufferInfo globalUboBufferInfo{ *ubos_.global, 0, ubo_size_.global };

		std::array descriptorWrites{
			 vk::WriteDescriptorSet{
				.dstSet = *sets_.global,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &globalUboBufferInfo
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Model UBO
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.model
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.model = std::move(sets.front());

		// Update
		vk::DescriptorBufferInfo modelUboBufferInfo{ *ubos_.model, 0, ubo_size_.model };

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.model,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &modelUboBufferInfo
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Lighting
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.lighting
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.lighting = std::move(sets.front());

		vk::DescriptorBufferInfo lightUboBufferInfo{ *ubos_.light, 0, ubo_size_.light };

		std::array<vk::DescriptorImageInfo, 4> gbufferInfos{
			vk::DescriptorImageInfo{
				.sampler = *geometry_buffers_.sampler,
				.imageView = *geometry_buffers_.albedo_mettalic_image_view,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			},
			vk::DescriptorImageInfo{
				.sampler = *geometry_buffers_.sampler,
				.imageView = *geometry_buffers_.normal_roughness_image_view,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			},
			vk::DescriptorImageInfo{
				.sampler = *geometry_buffers_.sampler,
				.imageView = *geometry_buffers_.height_ao_image_view,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			},
			vk::DescriptorImageInfo{
				.sampler = *geometry_buffers_.sampler,
				.imageView = *geometry_buffers_.coat_fuzz_image_view,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			},
		};

		vk::DescriptorImageInfo depthImageInfo{
			.sampler = *geometry_buffers_.sampler,
			.imageView = *depth_image_view_,
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
		};

		vk::DescriptorImageInfo shadowDepthOnlyInfo{
			.sampler = *shadow_sampler,
			.imageView = *shadow_image_view_,
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
		};

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &lightUboBufferInfo
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &gbufferInfos[0]
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &gbufferInfos[1]
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &gbufferInfos[2]
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 4,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &gbufferInfos[3]
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 5,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &depthImageInfo
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.lighting,
				.dstBinding = 6,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo = &shadowDepthOnlyInfo
			},
		};

		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Skybox
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.skybox
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.skybox = std::move(sets.front());

		vk::DescriptorBufferInfo skyboxUboBufferInfo{ *ubos_.skybox, 0, ubo_size_.skybox };

		std::array descriptorWrites{
			 vk::WriteDescriptorSet{
				.dstSet = *sets_.skybox,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &skyboxUboBufferInfo
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Cloth
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.cloth
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.cloth = std::move(sets.front());

		vk::DescriptorBufferInfo clothUbo{ *ubos_.cloth, 0, ubo_size_.cloth };
		vk::DescriptorBufferInfo positions(particle_manager_.ssbos_.position, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo normals(particle_manager_.ssbos_.normal, 0, VK_WHOLE_SIZE);

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &clothUbo
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &positions
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.cloth,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &normals
			},

		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Softbody
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.softbody
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.softbody = std::move(sets.front());

		vk::DescriptorBufferInfo softbodyUbo{ *ubos_.softbody, 0, ubo_size_.softbody };
		vk::DescriptorBufferInfo positions(particle_manager_.ssbos_.position, 0, VK_WHOLE_SIZE);
		vk::DescriptorBufferInfo normals(particle_manager_.ssbos_.normal, 0, VK_WHOLE_SIZE);

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.softbody,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &softbodyUbo
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.softbody,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &positions
			},
			vk::WriteDescriptorSet{
				.dstSet = *sets_.softbody,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &normals
			},

		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Skinned Model UBO
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.skinned_model
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.skinned_model = std::move(sets.front());

		// Update
		vk::DescriptorBufferInfo skinnedModelUboBufferInfo{ *ubos_.skinned_model, 0, ubo_size_.skinned_model };

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.skinned_model,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &skinnedModelUboBufferInfo
			},
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// Shadow
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.shadow_particle
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.shadow_particle = std::move(sets.front());

		vk::DescriptorBufferInfo positions(particle_manager_.ssbos_.position, 0, VK_WHOLE_SIZE);

		std::array descriptorWrites{
			vk::WriteDescriptorSet{
				.dstSet = *sets_.shadow_particle,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo = &positions
			},
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}

	// grid
	{
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool = *descriptor_pool_,
			.descriptorSetCount = 1,
			.pSetLayouts = &*set_layouts_.grid
		};

		auto sets = vk::raii::DescriptorSets{ context_.device_, allocInfo };
		sets_.grid = std::move(sets.front());

		vk::DescriptorBufferInfo gridUboBufferInfo{ *ubos_.grid, 0, ubo_size_.grid };

		std::array descriptorWrites{
			 vk::WriteDescriptorSet{
				.dstSet = *sets_.grid,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = vk::DescriptorType::eUniformBufferDynamic,
				.pBufferInfo = &gridUboBufferInfo
			}
		};
		context_.device_.updateDescriptorSets(descriptorWrites, {});
	}
}

void GraphicsPass::CreateGeometryBuffers()
{
	//RT0: albedo + metallic �� VK_FORMAT_R8G8B8A8_UNORM
	{
		vk::Format format = vk::Format::eR8G8B8A8Unorm;
		auto& image = geometry_buffers_.albedo_mettalic_image;
		auto& imageView = geometry_buffers_.albedo_mettalic_image_view;
		auto& memory = geometry_buffers_.albedo_mettalic_image_memory;
		geometry_buffers_.formats.push_back(format);
		vku::CreateImage(
			context_.physical_device_, context_.device_,
			swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height,
			1, vk::SampleCountFlagBits::e1,
			format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);
		imageView = vku::CreateImageView(context_.device_, image, format, vk::ImageAspectFlagBits::eColor, 1);
	}

	//RT1 : normal + roughness �� VK_FORMAT_R8G8B8A8_UNORM
	{
		vk::Format format = vk::Format::eR8G8B8A8Unorm;
		geometry_buffers_.formats.push_back(format);
		auto& image = geometry_buffers_.normal_roughness_image;
		auto& imageView = geometry_buffers_.normal_roughness_image_view;
		auto& memory = geometry_buffers_.normal_roughness_image_memory;
		vku::CreateImage(
			context_.physical_device_, context_.device_,
			swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height,
			1, vk::SampleCountFlagBits::e1,
			format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);
		imageView = vku::CreateImageView(context_.device_, image, format, vk::ImageAspectFlagBits::eColor, 1);
	}

	//RT2 : height + ao �� VK_FORMAT_R8G8_UNORM
	{
		vk::Format format = vk::Format::eR8G8Unorm;
		geometry_buffers_.formats.push_back(format);
		auto& image = geometry_buffers_.height_ao_image;
		auto& imageView = geometry_buffers_.height_ao_image_view;
		auto& memory = geometry_buffers_.height_ao_image_memory;
		vku::CreateImage(
			context_.physical_device_, context_.device_,
			swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height,
			1, vk::SampleCountFlagBits::e1,
			format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);
		imageView = vku::CreateImageView(context_.device_, image, format, vk::ImageAspectFlagBits::eColor, 1);
	}

	//RT3 : coat + coat_roughness + fuzz + fuzz_roughness �� VK_FORMAT_R8G8B8A8_UNORM
	{
		vk::Format format = vk::Format::eR8G8B8A8Unorm;
		geometry_buffers_.formats.push_back(format);
		auto& image = geometry_buffers_.coat_fuzz_image;
		auto& imageView = geometry_buffers_.coat_fuzz_image_view;
		auto& memory = geometry_buffers_.coat_fuzz_image_memory;
		vku::CreateImage(
			context_.physical_device_, context_.device_,
			swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height,
			1, vk::SampleCountFlagBits::e1,
			format, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, image, memory);
		imageView = vku::CreateImageView(context_.device_, image, format, vk::ImageAspectFlagBits::eColor, 1);
	}

	vk::SamplerCreateInfo samplerInfo{
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eNearest,
		.addressModeU = vk::SamplerAddressMode::eClampToEdge,
		.addressModeV = vk::SamplerAddressMode::eClampToEdge,
		.addressModeW = vk::SamplerAddressMode::eClampToEdge,
		.minLod = 0.0f,
		.maxLod = 0.0f
	};
	geometry_buffers_.sampler = vk::raii::Sampler(context_.device_, samplerInfo);
}

void GraphicsPass::CreateDepthResources() {
	vk::Format depthFormat = vku::FindDepthFormat(context_.physical_device_);

	depth_image_ = nullptr;
	depth_image_memory_ = nullptr;
	depth_image_view_ = nullptr;
	vku::CreateImage(context_.physical_device_, context_.device_, swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height, 1, msaa_samples_, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, depth_image_, depth_image_memory_);
	depth_image_view_ = vku::CreateImageView(context_.device_, depth_image_, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
}

void GraphicsPass::CreateShadowResources()
{
	vk::Format format = vk::Format::eD32Sfloat;

	vku::CreateImage(context_.physical_device_, context_.device_, shadow_extent_.width, shadow_extent_.height, 1, vk::SampleCountFlagBits::e1, format, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, shadow_image_, shadow_image_memory_);
	shadow_image_view_ = vku::CreateImageView(context_.device_, shadow_image_, format, vk::ImageAspectFlagBits::eDepth, 1);

	vk::SamplerCreateInfo samplerInfo{
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eNearest,
		.addressModeU = vk::SamplerAddressMode::eClampToEdge,
		.addressModeV = vk::SamplerAddressMode::eClampToEdge,
		.addressModeW = vk::SamplerAddressMode::eClampToEdge,
		.compareEnable = vk::True,
		.compareOp = vk::CompareOp::eLessOrEqual,
		.minLod = 0.0f,
		.maxLod = 0.0f,
	};
	shadow_sampler = vk::raii::Sampler(context_.device_, samplerInfo);
}

void GraphicsPass::CreateGraphicsPipelines()
{

	vk::Format depthFormat = vku::FindDepthFormat(context_.physical_device_);

	// Model
	{
		auto vertCode = vku::ReadFile("shaders/spv/model.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/model.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context_.device_, fragCode);

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

		// Vectex Input
		auto vdesc = Vertex::GetInputDescription(vku::VertexIncludeInfo{ true, true, true, false, false });

		vk::PipelineVertexInputStateCreateInfo vi{};
		vi.vertexBindingDescriptionCount = static_cast<uint32_t>(vdesc.bindings.size());
		vi.pVertexBindingDescriptions = vdesc.bindings.data();
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(vdesc.attributes.size());
		vi.pVertexAttributeDescriptions = vdesc.attributes.data();

		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};
		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = msaa_samples_,
			.sampleShadingEnable = vk::False
		};
		vk::PipelineDepthStencilStateCreateInfo ds{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		std::array<vk::PipelineColorBlendAttachmentState, 4> blendAtt{};
		for (auto& a : blendAtt) {
			a.colorWriteMask =
				vk::ColorComponentFlagBits::eR |
				vk::ColorComponentFlagBits::eG |
				vk::ColorComponentFlagBits::eB |
				vk::ColorComponentFlagBits::eA;
			a.blendEnable = vk::False;
		}

		vk::PipelineColorBlendStateCreateInfo cb{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = blendAtt.size(),
			.pAttachments = blendAtt.data()
		};

		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 3> setLayouts(
			*set_layouts_.global,
			*set_layouts_.model,
			*texture_manager_.set_layouts_.tex2d
		);
		vk::PipelineLayoutCreateInfo pi{
			.setLayoutCount = setLayouts.size(),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 0
		};
		pipeline_layouts_.model = vk::raii::PipelineLayout(context_.device_, pi);

		vk::PipelineRenderingCreateInfo renderingInfo{
			.colorAttachmentCount = static_cast<uint32_t>(geometry_buffers_.formats.size()),
			.pColorAttachmentFormats = geometry_buffers_.formats.data(),
			.depthAttachmentFormat = depthFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
		{
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = &cb,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.model,
				.renderPass = nullptr
			},
			{
				renderingInfo
			}
		};
		pipelines_.model_solid = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rs.polygonMode = vk::PolygonMode::eLine;
		rs.cullMode = vk::CullModeFlagBits::eNone;
		pipelines_.model_wireframe = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rs.polygonMode = vk::PolygonMode::ePoint;
		pipelines_.model_point = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// Lighting pipeline
	{
		auto vertCode = vku::ReadFile("shaders/spv/lighting.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/lighting.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context_.device_, fragCode);

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

		vk::PipelineVertexInputStateCreateInfo vi{};

		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};
		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eNone,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = msaa_samples_,
			.sampleShadingEnable = vk::False
		};

		vk::PipelineDepthStencilStateCreateInfo ds{};

		// color blend: attachment 1
		vk::PipelineColorBlendAttachmentState blendAtt{};
		blendAtt.blendEnable = vk::False;
		blendAtt.colorWriteMask =
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;

		vk::PipelineColorBlendStateCreateInfo cb{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = 1,
			.pAttachments = &blendAtt
		};
		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};

		std::array<vk::DescriptorSetLayout, 4> setLayouts{
			*set_layouts_.lighting,
			*set_layouts_.skybox,
			*texture_manager_.set_layouts_.tex2d,
			*texture_manager_.set_layouts_.tex_env
		};
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 0
		};
		pipeline_layouts_.lighting =
			vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::Format swapchainFormat = swapchain_.swapchain_surface_format_.format;
		vk::PipelineRenderingCreateInfo renderingInfo{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapchainFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
		{
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = &cb,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.lighting,
				.renderPass = nullptr
			},
			{
				renderingInfo
			}
		};

		pipelines_.lighting = vk::raii::Pipeline(
			context_.device_, nullptr,
			pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// Skybox
	{
		auto vertCode = vku::ReadFile("shaders/spv/skybox.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/skybox.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context_.device_, fragCode);

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

		auto vdesc = Vertex::GetInputDescription(vku::VertexIncludeInfo{ false, false, false, false, false });

		vk::PipelineVertexInputStateCreateInfo vi{};
		vi.vertexBindingDescriptionCount = static_cast<uint32_t>(vdesc.bindings.size());
		vi.pVertexBindingDescriptions = vdesc.bindings.data();
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(vdesc.attributes.size());
		vi.pVertexAttributeDescriptions = vdesc.attributes.data();

		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};
		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = msaa_samples_,
			.sampleShadingEnable = vk::False
		};
		vk::PipelineDepthStencilStateCreateInfo ds{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::False,
			.depthCompareOp = vk::CompareOp::eLessOrEqual,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		// color blend: attachment 1
		vk::PipelineColorBlendAttachmentState blendAtt{};
		blendAtt.blendEnable = vk::False;
		blendAtt.colorWriteMask =
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;

		vk::PipelineColorBlendStateCreateInfo cb{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = 1,
			.pAttachments = &blendAtt
		};

		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 3> setLayouts(
			*set_layouts_.global,
			*set_layouts_.skybox,
			*texture_manager_.set_layouts_.tex_env);
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = setLayouts.size(), .pSetLayouts = setLayouts.data(), .pushConstantRangeCount = 0 };
		pipeline_layouts_.skybox = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::PipelineRasterizationStateCreateInfo rasterizer{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::Format swapchainFormat = swapchain_.swapchain_surface_format_.format;

		vk::PipelineRenderingCreateInfo renderingInfo{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapchainFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
		{
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = &cb,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.skybox,
				.renderPass = nullptr
			},
			{
				renderingInfo
			}
		};

	}

	// Cloth
	{
		auto vertCode = vku::ReadFile("shaders/spv/cloth.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/cloth.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context_.device_, fragCode);

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
		vk::PipelineVertexInputStateCreateInfo vi{};

		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};
		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eNone,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = msaa_samples_,
			.sampleShadingEnable = vk::False
		};
		vk::PipelineDepthStencilStateCreateInfo ds{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		std::array<vk::PipelineColorBlendAttachmentState, 4> blendAtt{};
		for (auto& a : blendAtt) {
			a.colorWriteMask =
				vk::ColorComponentFlagBits::eR |
				vk::ColorComponentFlagBits::eG |
				vk::ColorComponentFlagBits::eB |
				vk::ColorComponentFlagBits::eA;
			a.blendEnable = vk::False;
		}

		vk::PipelineColorBlendStateCreateInfo cb{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = blendAtt.size(),
			.pAttachments = blendAtt.data()
		};

		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 3> setLayouts(
			*set_layouts_.global,
			*set_layouts_.cloth,
			*texture_manager_.set_layouts_.tex2d);

		vk::PushConstantRange pcRange{
			.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			.offset = 0,
			.size = static_cast<uint32_t>(sizeof(PushConstant::ClothRender))
		};
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = setLayouts.size(),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcRange
		};
		pipeline_layouts_.cloth = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::PipelineRenderingCreateInfo renderingInfo{
			.colorAttachmentCount = static_cast<uint32_t>(geometry_buffers_.formats.size()),
			.pColorAttachmentFormats = geometry_buffers_.formats.data(),
			.depthAttachmentFormat = depthFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
		{
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = &cb,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.cloth,
				.renderPass = nullptr
			},
			{
				renderingInfo
			}
		};

		pipelines_.cloth_solid = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rs.polygonMode = vk::PolygonMode::eLine;
		pipelines_.cloth_wireframe = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rs.polygonMode = vk::PolygonMode::ePoint;
		pipelines_.cloth_point = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

	}

	// Softbody
	{
		// Shader
		auto vertCode = vku::ReadFile("shaders/spv/softbody.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/softbody.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context_.device_, fragCode);

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
		vk::PipelineVertexInputStateCreateInfo vi{};

		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};
		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eNone,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = msaa_samples_,
			.sampleShadingEnable = vk::False
		};
		vk::PipelineDepthStencilStateCreateInfo ds{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		std::array<vk::PipelineColorBlendAttachmentState, 4> blendAtt{};
		for (auto& a : blendAtt) {
			a.colorWriteMask =
				vk::ColorComponentFlagBits::eR |
				vk::ColorComponentFlagBits::eG |
				vk::ColorComponentFlagBits::eB |
				vk::ColorComponentFlagBits::eA;
			a.blendEnable = vk::False;
		}

		vk::PipelineColorBlendStateCreateInfo cb{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = blendAtt.size(),
			.pAttachments = blendAtt.data()
		};

		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 2> setLayouts(
			*set_layouts_.global,
			*set_layouts_.softbody
		);

		vk::PushConstantRange pcRange{
			.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			.offset = 0,
			.size = static_cast<uint32_t>(sizeof(PushConstant::SoftBody))
		};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = setLayouts.size(),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcRange
		};
		pipeline_layouts_.softbody = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::PipelineRenderingCreateInfo renderingInfo{
			.colorAttachmentCount = static_cast<uint32_t>(geometry_buffers_.formats.size()),
			.pColorAttachmentFormats = geometry_buffers_.formats.data(),
			.depthAttachmentFormat = depthFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
		{
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = &cb,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.softbody,
				.renderPass = nullptr
			},
			{
				renderingInfo
			}
		};
		pipelines_.softbody_solid = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rs.polygonMode = vk::PolygonMode::eLine;
		pipelines_.softbody_wireframe = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rs.polygonMode = vk::PolygonMode::ePoint;
		pipelines_.softbody_point = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// Skinned Model
	{
		// Shader
		auto vertCode = vku::ReadFile("shaders/spv/skinned_model.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/skinned_model.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context_.device_, fragCode);

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

		// Vectex Input
		auto vdesc = Vertex::GetInputDescription(vku::VertexIncludeInfo{ true, true, true, true, true });

		vk::PipelineVertexInputStateCreateInfo vi{};
		vi.vertexBindingDescriptionCount = static_cast<uint32_t>(vdesc.bindings.size());
		vi.pVertexBindingDescriptions = vdesc.bindings.data();
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(vdesc.attributes.size());
		vi.pVertexAttributeDescriptions = vdesc.attributes.data();

		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};
		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f
		};

		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = msaa_samples_,
			.sampleShadingEnable = vk::False
		};
		vk::PipelineDepthStencilStateCreateInfo ds{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		std::array<vk::PipelineColorBlendAttachmentState, 4> blendAtt{};
		for (auto& a : blendAtt) {
			a.colorWriteMask =
				vk::ColorComponentFlagBits::eR |
				vk::ColorComponentFlagBits::eG |
				vk::ColorComponentFlagBits::eB |
				vk::ColorComponentFlagBits::eA;
			a.blendEnable = vk::False;
		}

		vk::PipelineColorBlendStateCreateInfo cb{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = blendAtt.size(),
			.pAttachments = blendAtt.data()
		};

		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 4> setLayouts(
			*set_layouts_.global,
			*set_layouts_.model,
			*texture_manager_.set_layouts_.tex2d,
			*set_layouts_.skinned_model
		);
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = setLayouts.size(),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 0
		};
		pipeline_layouts_.skinned_model = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::PipelineRenderingCreateInfo renderingInfo{
			.colorAttachmentCount = static_cast<uint32_t>(geometry_buffers_.formats.size()),
			.pColorAttachmentFormats = geometry_buffers_.formats.data(),
			.depthAttachmentFormat = depthFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
		{
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = &cb,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.skinned_model,
				.renderPass = nullptr
			},
			{
				renderingInfo
			}
		};
		pipelines_.skinned_model_solid = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rs.polygonMode = vk::PolygonMode::eLine;
		pipelines_.skinned_model_wireframe = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());

		rs.polygonMode = vk::PolygonMode::ePoint;
		pipelines_.skinned_model_point = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// Debug Capsule
	{
		// Shader
		auto vertCode = vku::ReadFile("shaders/spv/debug_capsule.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/debug_capsule.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context_.device_, fragCode);

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

		// Vectex Input
		auto vdesc = Vertex::GetInputDescription(vku::VertexIncludeInfo{ true, true, false, false, false });

		vk::PipelineVertexInputStateCreateInfo vi{};
		vi.vertexBindingDescriptionCount = static_cast<uint32_t>(vdesc.bindings.size());
		vi.pVertexBindingDescriptions = vdesc.bindings.data();
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(vdesc.attributes.size());
		vi.pVertexAttributeDescriptions = vdesc.attributes.data();


		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};
		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eBack,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,
			.lineWidth = 1.0f,
		};

		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = msaa_samples_,
			.sampleShadingEnable = vk::False
		};
		vk::PipelineDepthStencilStateCreateInfo ds{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		std::array<vk::PipelineColorBlendAttachmentState, 4> blendAtt{};
		for (auto& a : blendAtt) {
			a.colorWriteMask =
				vk::ColorComponentFlagBits::eR |
				vk::ColorComponentFlagBits::eG |
				vk::ColorComponentFlagBits::eB |
				vk::ColorComponentFlagBits::eA;
			a.blendEnable = vk::False;
		}

		vk::PipelineColorBlendStateCreateInfo cb{
			.logicOpEnable = vk::False,
			.logicOp = vk::LogicOp::eCopy,
			.attachmentCount = blendAtt.size(),
			.pAttachments = blendAtt.data()
		};

		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 1> setLayouts(
			*set_layouts_.global
		);

		vk::PushConstantRange pcRange{
			.stageFlags = vk::ShaderStageFlagBits::eVertex,
			.offset = 0,
			.size = static_cast<uint32_t>(sizeof(glm::mat4) + sizeof(glm::vec4))
		};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = setLayouts.size(),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcRange
		};
		pipeline_layouts_.debug_capsule = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::PipelineRenderingCreateInfo renderingInfo{
			.colorAttachmentCount = static_cast<uint32_t>(geometry_buffers_.formats.size()),
			.pColorAttachmentFormats = geometry_buffers_.formats.data(),
			.depthAttachmentFormat = depthFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain =
		{
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = &cb,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.debug_capsule,
				.renderPass = nullptr
			},
			{
				renderingInfo
			}
		};
		rs.polygonMode = vk::PolygonMode::eLine;
		pipelines_.debug_capsule = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// shadow_model
	{
		auto vertCode = vku::ReadFile("shaders/spv/shadow_model.vert.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);

		vk::PipelineShaderStageCreateInfo vertStage{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = *vertModule,
			.pName = "main"
		};
		std::array<vk::PipelineShaderStageCreateInfo, 1> stages{ vertStage };

		// Vectex Input
		auto vdesc = Vertex::GetInputDescription(vku::VertexIncludeInfo{ false, false, false, false, false });

		vk::PipelineVertexInputStateCreateInfo vi{};
		vi.vertexBindingDescriptionCount = static_cast<uint32_t>(vdesc.bindings.size());
		vi.pVertexBindingDescriptions = vdesc.bindings.data();
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(vdesc.attributes.size());
		vi.pVertexAttributeDescriptions = vdesc.attributes.data();

		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};
		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eNone,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::True,
			.lineWidth = 1.0f,
		};

		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = msaa_samples_,
			.sampleShadingEnable = vk::False
		};
		vk::PipelineDepthStencilStateCreateInfo ds{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLessOrEqual,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False,
		};

		//std::array<vk::PipelineColorBlendAttachmentState, 4> blendAtt{};
		//for (auto& a : blendAtt) {
		//	a.colorWriteMask =
		//		vk::ColorComponentFlagBits::eR |
		//		vk::ColorComponentFlagBits::eG |
		//		vk::ColorComponentFlagBits::eB |
		//		vk::ColorComponentFlagBits::eA;
		//	a.blendEnable = vk::False;
		//}

		//vk::PipelineColorBlendStateCreateInfo cb{
		//	.logicOpEnable = vk::False,
		//	.logicOp = vk::LogicOp::eCopy,
		//	.attachmentCount = blendAtt.size(),
		//	.pAttachments = blendAtt.data()
		//};

		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};
		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 1> setLayouts(
			*set_layouts_.model
		);

		vk::PushConstantRange pcRange{
			.stageFlags = vk::ShaderStageFlagBits::eVertex,
			.offset = 0,
			.size = static_cast<uint32_t>(sizeof(ShadowMap))
		};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = setLayouts.size(),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcRange
		};
		pipeline_layouts_.shadow_model = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::PipelineRenderingCreateInfo renderingInfo{
			.colorAttachmentCount = 0,
			.pColorAttachmentFormats = nullptr,
			.depthAttachmentFormat = depthFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = nullptr,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.shadow_model,
				.renderPass = nullptr
			},
			{
				renderingInfo
			}
		};
		pipelines_.shadow_model = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// shadow_particle
	{
		auto vertCode = vku::ReadFile("shaders/spv/shadow_particle.vert.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);

		vk::PipelineShaderStageCreateInfo vertStage{
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = *vertModule,
			.pName = "main"
		};
		std::array<vk::PipelineShaderStageCreateInfo, 1> stages{ vertStage };

		// Vectex Input
		auto vdesc = Vertex::GetInputDescription(vku::VertexIncludeInfo{ false, false, false, false, false });

		vk::PipelineVertexInputStateCreateInfo vi{};
		vi.vertexBindingDescriptionCount = static_cast<uint32_t>(vdesc.bindings.size());
		vi.pVertexBindingDescriptions = vdesc.bindings.data();
		vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(vdesc.attributes.size());
		vi.pVertexAttributeDescriptions = vdesc.attributes.data();

		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};
		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eNone,
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::True,
			.lineWidth = 1.0f,
		};

		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = msaa_samples_,
			.sampleShadingEnable = vk::False
		};
		vk::PipelineDepthStencilStateCreateInfo ds{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLessOrEqual,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False,
		};

		//std::array<vk::PipelineColorBlendAttachmentState, 4> blendAtt{};
		//for (auto& a : blendAtt) {
		//	a.colorWriteMask =
		//		vk::ColorComponentFlagBits::eR |
		//		vk::ColorComponentFlagBits::eG |
		//		vk::ColorComponentFlagBits::eB |
		//		vk::ColorComponentFlagBits::eA;
		//	a.blendEnable = vk::False;
		//}

		//vk::PipelineColorBlendStateCreateInfo cb{
		//	.logicOpEnable = vk::False,
		//	.logicOp = vk::LogicOp::eCopy,
		//	.attachmentCount = blendAtt.size(),
		//	.pAttachments = blendAtt.data()
		//};

		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 1> setLayouts(
			*set_layouts_.shadow_particle
		);

		vk::PushConstantRange pcRange{
			.stageFlags = vk::ShaderStageFlagBits::eVertex,
			.offset = 0,
			.size = static_cast<uint32_t>(sizeof(ShadowMap))
		};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = setLayouts.size(),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pcRange
		};
		pipeline_layouts_.shadow_particle = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::PipelineRenderingCreateInfo renderingInfo{
			.colorAttachmentCount = 0,
			.pColorAttachmentFormats = nullptr,
			.depthAttachmentFormat = depthFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		// Pipeline
		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = nullptr,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.shadow_particle,
				.renderPass = nullptr
			},
			{
				renderingInfo
			}
		};
		pipelines_.shadow_particle = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	// infinite_grid
	{
		auto vertCode = vku::ReadFile("shaders/spv/infinite_grid.vert.spv");
		auto fragCode = vku::ReadFile("shaders/spv/infinite_grid.frag.spv");

		vk::raii::ShaderModule vertModule = vku::CreateShaderModule(context_.device_, vertCode);
		vk::raii::ShaderModule fragModule = vku::CreateShaderModule(context_.device_, fragCode);

		std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {
			vk::PipelineShaderStageCreateInfo{
				.stage = vk::ShaderStageFlagBits::eVertex,
				.module = *vertModule,
				.pName = "main",
			},
			vk::PipelineShaderStageCreateInfo{
				.stage = vk::ShaderStageFlagBits::eFragment,
				.module = *fragModule,
				.pName = "main",
			}
		};

		// No vertex buffers (fullscreen quad generated from gl_VertexIndex)
		vk::PipelineVertexInputStateCreateInfo vi{};

		vk::PipelineInputAssemblyStateCreateInfo ia{
			.topology = vk::PrimitiveTopology::eTriangleList,
			.primitiveRestartEnable = vk::False
		};

		vk::PipelineViewportStateCreateInfo vs{
			.viewportCount = 1,
			.scissorCount = 1
		};

		// IMPORTANT: match this to the actual render target sample count.
		vk::PipelineMultisampleStateCreateInfo ms{
			.rasterizationSamples = vk::SampleCountFlagBits::e1
		};

		vk::PipelineRasterizationStateCreateInfo rs{
			.depthClampEnable = vk::False,
			.rasterizerDiscardEnable = vk::False,
			.polygonMode = vk::PolygonMode::eFill,
			.cullMode = vk::CullModeFlagBits::eNone,  // grid: double sided
			.frontFace = vk::FrontFace::eCounterClockwise,
			.depthBiasEnable = vk::False,                     // not shadow
			.lineWidth = 1.0f
		};

		// Depth test ON, depth write OFF (draw after opaque)
		vk::PipelineDepthStencilStateCreateInfo ds{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::False,
			.depthCompareOp = vk::CompareOp::eLessOrEqual,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False
		};

		// Alpha blending for fade
		vk::PipelineColorBlendAttachmentState blendAtt{};
		blendAtt.colorWriteMask =
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;

		//blendAtt.blendEnable = vk::True;
		//blendAtt.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
		//blendAtt.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		//blendAtt.colorBlendOp = vk::BlendOp::eAdd;
		//blendAtt.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		//blendAtt.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		//blendAtt.alphaBlendOp = vk::BlendOp::eAdd;

		vk::PipelineColorBlendStateCreateInfo cb{
			.logicOpEnable = vk::False,
			.attachmentCount = 1,
			.pAttachments = &blendAtt
		};

		std::array<vk::DynamicState, 2> dynStates = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dyn{
			.dynamicStateCount = (uint32_t)dynStates.size(),
			.pDynamicStates = dynStates.data()
		};

		// Pipeline Layout
		std::array<vk::DescriptorSetLayout, 1> setLayouts(
			*set_layouts_.grid
		);

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
			.setLayoutCount = setLayouts.size(),
			.pSetLayouts = setLayouts.data()
		};
		pipeline_layouts_.infinite_grid = vk::raii::PipelineLayout(context_.device_, pipelineLayoutInfo);

		vk::Format colorFormat = swapchain_.swapchain_surface_format_.format;
		vk::Format depthFormat = vku::FindDepthFormat(context_.physical_device_);

		vk::PipelineRenderingCreateInfo ri{
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &colorFormat,
			.depthAttachmentFormat = depthFormat,
			.stencilAttachmentFormat = vk::Format::eUndefined
		};

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			{
				.stageCount = stages.size(),
				.pStages = stages.data(),
				.pVertexInputState = &vi,
				.pInputAssemblyState = &ia,
				.pViewportState = &vs,
				.pRasterizationState = &rs,
				.pMultisampleState = &ms,
				.pDepthStencilState = &ds,
				.pColorBlendState = &cb,
				.pDynamicState = &dyn,
				.layout = pipeline_layouts_.infinite_grid,
				.renderPass = nullptr
			},
			{
				ri
			}
		};
		pipelines_.infinite_grid = vk::raii::Pipeline(context_.device_, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}
}

void GraphicsPass::ShadowDepthOnlyPass(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame)
{
	const bool first = first_frame_;

	const vk::ImageLayout oldLayout = first
		? vk::ImageLayout::eUndefined
		: vk::ImageLayout::eShaderReadOnlyOptimal;

	const vk::PipelineStageFlags2 srcStage = first
		? vk::PipelineStageFlagBits2::eTopOfPipe
		: vk::PipelineStageFlagBits2::eFragmentShader;

	const vk::AccessFlags2 srcAccess = first
		? vk::AccessFlags2{}
	: vk::AccessFlagBits2::eShaderSampledRead;

	vku::TransitionImageLayoutCustom(
		shadow_image_,
		cmd,
		oldLayout,
		vk::ImageLayout::eDepthAttachmentOptimal,
		srcAccess,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		srcStage,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);

	vk::RenderingAttachmentInfo depthAttachmentInfo = {
		.imageView = shadow_image_view_,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore, // IMPORTANT: must store for sampling later
		.clearValue = vk::ClearDepthStencilValue(1.0f, 0),
	};

	vk::RenderingInfo renderingInfo = {
		.renderArea = {
			.offset = { 0, 0 },
			.extent = shadow_extent_,
		},
		.layerCount = 1,
		.pDepthAttachment = &depthAttachmentInfo
	};

	cmd.beginRendering(renderingInfo);

	vk::Viewport vp(
		0.0f,
		0.0f,
		shadow_extent_.width,
		shadow_extent_.height,
		0.0f, 1.0f
	);

	cmd.setViewport(0, vp);
	cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), shadow_extent_));

	//cmd.setDepthBias(/*constant=*/1.25f, /*clamp=*/0.0f, /*slope=*/1.75f);

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.shadow_model);

	shadow_map.light_view_proj = ubo_datas_.light.light_view_proj;
	shadow_map.is_vertex_ssbo = 0u;
	cmd.pushConstants<ShadowMap>(
		*pipeline_layouts_.shadow_model,
		vk::ShaderStageFlagBits::eVertex,
		0,
		shadow_map);

	const uint32_t baseObjectOffset = static_cast<uint32_t>(currentFrame * ubo_size_.model * model_manager_.kMaxModels);
	for (uint32_t i = 0; i < model_manager_.models_.size(); i++)
	{
		auto& model = *model_manager_.models_[i];

		if (!model.render_) continue;

		uint32_t objectOffset = baseObjectOffset + i * static_cast<uint32_t>(ubo_size_.model);

		// Model set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.shadow_model,
			0,
			{ *sets_.model },
			{ objectOffset }
		);

		cmd.bindVertexBuffers(0, { model.model_loader_->mesh_.vertex_buffer }, { 0 });
		cmd.bindIndexBuffer(*model.model_loader_->mesh_.index_buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(model.model_loader_->mesh_.indices_count, 1, 0, 0, 0);
	}

	{

		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.shadow_particle);
		shadow_map.is_vertex_ssbo = 1u;
		cmd.pushConstants<ShadowMap>(
			*pipeline_layouts_.shadow_particle,
			vk::ShaderStageFlagBits::eVertex,
			0,
			shadow_map);

		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.shadow_particle,
			0,
			{ *sets_.shadow_particle },
			{  }
		);

		cmd.bindIndexBuffer(*particle_manager_.index_buffer_, 0, vk::IndexType::eUint32);
		uint32_t totalIndices = particle_manager_.num_cloth_indices_ + particle_manager_.num_softbody_indices_;
		cmd.drawIndexed(totalIndices, 1, 0, 0, 0);
	}

	cmd.endRendering();

	{
		vku::TransitionImageLayoutCustom(
			shadow_image_,
			cmd,
			vk::ImageLayout::eDepthAttachmentOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
			vk::AccessFlagBits2::eShaderSampledRead,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
			vk::PipelineStageFlagBits2::eFragmentShader,
			vk::ImageAspectFlagBits::eDepth
		);
	}
}

void GraphicsPass::PreMainRenderPass(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame)
{

	auto toShaderWrite = [&](vk::raii::Image& img) {
		vku::TransitionImageLayoutCustom(
			img,
			cmd,
			first_frame_ ? vk::ImageLayout::eUndefined
			: vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::ImageLayout::eColorAttachmentOptimal,
			{},
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::PipelineStageFlagBits2::eTopOfPipe,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::ImageAspectFlagBits::eColor
		);
		};

	// G-buffer: Undefined/ShaderReadOnly �� ColorAttachmentOptimal
	toShaderWrite(geometry_buffers_.albedo_mettalic_image);
	toShaderWrite(geometry_buffers_.normal_roughness_image);
	toShaderWrite(geometry_buffers_.height_ao_image);
	toShaderWrite(geometry_buffers_.coat_fuzz_image);

	vku::TransitionImageLayoutCustom(
		depth_image_,
		cmd,
		first_frame_ ? vk::ImageLayout::eUndefined
		: vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::ImageLayout::eDepthAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);

	std::array<vk::RenderingAttachmentInfo, 4> gbufferAttachments;

	auto renderingAttachmentInfo = [&](vk::raii::ImageView& imageView, vk::ClearValue clearColor) {
		return vk::RenderingAttachmentInfo{
			.imageView = *imageView,
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.resolveMode = {},
			.resolveImageView = {},
			.resolveImageLayout = {},
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor
		};
		};
	vk::ClearValue clearColor0 = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f); // albedo+metal
	gbufferAttachments[0] = renderingAttachmentInfo(geometry_buffers_.albedo_mettalic_image_view, clearColor0);
	vk::ClearValue clearColor1 = vk::ClearColorValue(0.5f, 0.5f, 1.0f, 1.0f); // normal default (0,0,1)
	gbufferAttachments[1] = renderingAttachmentInfo(geometry_buffers_.normal_roughness_image_view, clearColor1);
	vk::ClearValue clearColor2 = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f); // height+AO
	gbufferAttachments[2] = renderingAttachmentInfo(geometry_buffers_.height_ao_image_view, clearColor2);
	vk::ClearValue clearColor3 = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f); // coat+fuzz
	gbufferAttachments[3] = renderingAttachmentInfo(geometry_buffers_.coat_fuzz_image_view, clearColor3);

	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
	// Depth attachment
	vk::RenderingAttachmentInfo depthAttachmentInfo = {
		.imageView = depth_image_view_,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eDontCare,
		.clearValue = clearDepth
	};
	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = { 0, 0 },
		.extent = swapchain_.swapchain_extent_ },
		.layerCount = 1,
		.colorAttachmentCount = static_cast<uint32_t>(gbufferAttachments.size()),
		.pColorAttachments = gbufferAttachments.data(),
		.pDepthAttachment = &depthAttachmentInfo
	};

	cmd.beginRendering(renderingInfo);
}

void GraphicsPass::MainRenderPass(const vk::raii::CommandBuffer& cmd, uint32_t currentFrame)
{

	auto& pm = particle_manager_;

	vk::Viewport vp(
		0.0f,
		0.0f,
		static_cast<float>(swapchain_.swapchain_extent_.width),
		static_cast<float>(swapchain_.swapchain_extent_.height),
		0.0f, 1.0f
	);
	cmd.setViewport(0, vp);
	cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain_.swapchain_extent_));

	uint32_t globalOffset = static_cast<uint32_t>(currentFrame * ubo_size_.global);
	const uint32_t baseObjectOffset = static_cast<uint32_t>(currentFrame * ubo_size_.model * model_manager_.kMaxModels);

	// Model
	{
		for (uint32_t i = 0; i < model_manager_.models_.size(); i++)
		{
			auto& model = *model_manager_.models_[i];

			if (!model.render_) continue;

			uint32_t objectOffset = baseObjectOffset + i * static_cast<uint32_t>(ubo_size_.model);

			if (model.model_type_ == ModelType::SHAPE)
			{
				if (polygon_mode_ == vku::PolygonMode::SOLID)
					cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.model_solid);
				else if (polygon_mode_ == vku::PolygonMode::WIREFRAME)
					cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.model_wireframe);
				else if (polygon_mode_ == vku::PolygonMode::POINT)
					cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.model_point);

				// Global Set
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipeline_layouts_.model,
					0,
					{ *sets_.global },
					{ globalOffset }
				);

				// Model set
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipeline_layouts_.model,
					1,
					{ *sets_.model },
					{ objectOffset }
				);

				// tex2D
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipeline_layouts_.model,
					2,
					{ *texture_manager_.sets_.tex2d },
					{ }
				);

				cmd.bindVertexBuffers(0, { model.model_loader_->mesh_.vertex_buffer }, { 0 });
				cmd.bindIndexBuffer(*model.model_loader_->mesh_.index_buffer, 0, vk::IndexType::eUint32);
				cmd.drawIndexed(model.model_loader_->mesh_.indices_count, 1, 0, 0, 0);
			}
			else if (model.model_type_ == ModelType::SKINNED)
			{
				if (polygon_mode_ == vku::PolygonMode::SOLID)
					cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.skinned_model_solid);
				else if (polygon_mode_ == vku::PolygonMode::WIREFRAME)
					cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.skinned_model_wireframe);
				else if (polygon_mode_ == vku::PolygonMode::POINT)
					cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.skinned_model_point);

				// Global Set
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipeline_layouts_.skinned_model,
					0,
					{ *sets_.global },
					{ globalOffset }
				);

				// Model set
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipeline_layouts_.skinned_model,
					1,
					{ *sets_.model },
					{ objectOffset }
				);

				// tex2D
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipeline_layouts_.skinned_model,
					2,
					{ *texture_manager_.sets_.tex2d },
					{ }
				);

				// Skinned
				const uint32_t skinnedModelOff = static_cast<uint32_t>(currentFrame * ubo_size_.skinned_model);
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipeline_layouts_.skinned_model,
					3,
					{ *sets_.skinned_model },
					{ skinnedModelOff }
				);

				cmd.bindVertexBuffers(0, { model.model_loader_->mesh_.vertex_buffer }, { 0 });
				cmd.bindIndexBuffer(*model.model_loader_->mesh_.index_buffer, 0, vk::IndexType::eUint32);
				cmd.drawIndexed(model.model_loader_->mesh_.indices_count, 1, 0, 0, 0);

				// Debug Capsule
				if (!model.capsule_collision_render_) continue;

				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.debug_capsule);

				// Global Set
				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipeline_layouts_.debug_capsule,
					0,
					{ *sets_.global },
					{ globalOffset }
				);

				auto& debugCapsuleModel = model_manager_.debug_capsule_;

				cmd.bindVertexBuffers(0, { debugCapsuleModel->model_loader_->mesh_.vertex_buffer }, { 0 });
				cmd.bindIndexBuffer(*debugCapsuleModel->model_loader_->mesh_.index_buffer, 0, vk::IndexType::eUint32);

				for (const auto& inst : model_manager_.models_[model_manager_.models_.size() - 1]->capsule_colliders_) {
					glm::vec3 p0 = inst.p0;
					glm::vec3 p1 = inst.p1;
					float r = inst.radius;

					glm::vec3 center = 0.5f * (p0 + p1);
					glm::vec3 seg = p1 - p0;
					float len = glm::length(seg);
					if (len < 1e-4f) continue;

					glm::vec3 dir = seg / len;
					glm::quat q = glm::rotation(glm::vec3(0, 1, 0), dir);

					glm::mat4 R = glm::mat4_cast(q);
					glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(r, len, r));
					//glm::mat4 T = glm::translate(glm::mat4(1.0f), center);
					//glm::mat4 T = glm::translate(glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f)), center);
					glm::mat4 T = glm::translate(model_manager_.models_[model_manager_.models_.size() - 1]->world_, center);
					glm::mat4 M = T * R * S;

					struct DebugPushConst {
						glm::mat4 model;
						glm::vec4 color;
					} pc;

					pc.model = M;
					pc.color = glm::vec4(1, 0, 0, 1);

					cmd.pushConstants<DebugPushConst>(
						*pipeline_layouts_.debug_capsule,
						vk::ShaderStageFlagBits::eVertex,
						0,
						pc);
					cmd.drawIndexed(debugCapsuleModel->model_loader_->mesh_.indices_count, 1, 0, 0, 0);
				}
			}

		}
	}

	// Cloth
	{
		if (polygon_mode_ == vku::PolygonMode::WIREFRAME)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.cloth_wireframe);
		else if (polygon_mode_ == vku::PolygonMode::POINT)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.cloth_point);
		else
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.cloth_solid);

		// Global Set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.cloth,
			0,
			{ *sets_.global },
			{ globalOffset }
		);

		// Tex
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.cloth,
			2,
			{ *texture_manager_.sets_.tex2d },
			{ }
		);

		cmd.bindIndexBuffer(*pm.index_buffer_, 0, vk::IndexType::eUint32);

		const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * pm.clothes_.size() * ubo_size_.cloth);
		for (uint32_t i = 0; i < pm.clothes_.size(); i++)
		{
			auto& cloth = pm.clothes_[i];

			if (!cloth.render) continue;

			const uint32_t dst = baseOffset + ubo_size_.cloth * i;
			cmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				pipeline_layouts_.cloth,
				1,
				{ *sets_.cloth },
				{ dst }
			);

			push_constants_.cloth_render.color = cloth.color;
			push_constants_.cloth_render.nx1 = cloth.nx1;
			push_constants_.cloth_render.ny1 = cloth.ny1;
			push_constants_.cloth_render.offset_particle = cloth.offset_particle;
			cmd.pushConstants<PushConstant::ClothRender>(
				*pipeline_layouts_.cloth,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				0,
				push_constants_.cloth_render
			);

			cmd.drawIndexed(cloth.num_indices, 1, cloth.offset_indices, 0, 0);
		}
	}

	// Softbody
	{

		if (polygon_mode_ == vku::PolygonMode::WIREFRAME)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.softbody_wireframe);
		else if (polygon_mode_ == vku::PolygonMode::POINT)
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.softbody_point);
		else
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.softbody_solid);

		// Global Set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.softbody,
			0,
			{ *sets_.global },
			{ globalOffset }
		);

		cmd.bindIndexBuffer(*pm.index_buffer_, 0, vk::IndexType::eUint32);

		const uint32_t baseOffset = static_cast<uint32_t>(currentFrame * pm.softbodies_.size() * ubo_size_.softbody);

		for (uint32_t i = 0; i < pm.softbodies_.size(); i++)
		{
			auto& softbody = pm.softbodies_[i];

			if (!softbody.render) continue;

			const uint32_t offset = baseOffset + i * ubo_size_.softbody;
			cmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				pipeline_layouts_.softbody,
				1,
				{ *sets_.softbody },
				{ offset }
			);

			push_constants_.softbody.color = pm.softbodies_[i].color;
			cmd.pushConstants<PushConstant::SoftBody>(
				*pipeline_layouts_.softbody,
				vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
				0,
				push_constants_.softbody
			);

			cmd.drawIndexed(pm.softbodies_[i].num_indices, 1, pm.softbodies_[i].offset_indices, 0, 0);
		}
	}

	cmd.endRendering();
}

void GraphicsPass::PostMainRenderPass(const vk::raii::CommandBuffer& cmd, uint32_t imageIndex)
{
	auto toShaderRead = [&](vk::raii::Image& img) {
		vku::TransitionImageLayoutCustom(
			img,
			cmd,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::AccessFlagBits2::eShaderRead,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eFragmentShader,
			vk::ImageAspectFlagBits::eColor
		);
		};
	toShaderRead(geometry_buffers_.albedo_mettalic_image);
	toShaderRead(geometry_buffers_.normal_roughness_image);
	toShaderRead(geometry_buffers_.height_ao_image);
	toShaderRead(geometry_buffers_.coat_fuzz_image);

	vku::TransitionImageLayoutCustom(
		depth_image_,
		cmd,
		vk::ImageLayout::eDepthAttachmentOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::AccessFlagBits2::eShaderRead,
		vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eDepth
	);

	vku::TransitionImageLayout(
		swapchain_.swapchain_images_[imageIndex],
		cmd,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput
	);
}

void GraphicsPass::LightingPass(const vk::raii::CommandBuffer& cmd, uint32_t imageIndex, uint32_t currentFrame)
{
	vk::ClearValue clearColor = vk::ClearColorValue(
		0.0f, 0.0f, 0.0f, 1.0f);

	vk::RenderingAttachmentInfo colorAttachmentInfo{
		.imageView = swapchain_.swapchain_image_views_[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.resolveMode = {},
		.resolveImageView = {},
		.resolveImageLayout = {},
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor
	};

	vk::RenderingInfo lightingRenderingInfo{
		.renderArea = { {0, 0}, swapchain_.swapchain_extent_ },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachmentInfo,
		.pDepthAttachment = nullptr
	};

	cmd.beginRendering(lightingRenderingInfo);

	vk::Viewport vp(
		0.0f,
		0.0f,
		static_cast<float>(swapchain_.swapchain_extent_.width),
		static_cast<float>(swapchain_.swapchain_extent_.height),
		0.0f, 1.0f
	);
	cmd.setViewport(0, vp);
	cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain_.swapchain_extent_));

	// --- lighting pipeline bind ---
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.lighting);

	uint32_t lightOffset = static_cast<uint32_t>(currentFrame * ubo_size_.light);
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		pipeline_layouts_.lighting,
		0,
		{ *sets_.lighting },
		{ lightOffset }
	);

	uint32_t skyboxOffset = static_cast<uint32_t>(currentFrame * ubo_size_.skybox);
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		pipeline_layouts_.lighting,
		1,
		{ *sets_.skybox },
		{ skyboxOffset }
	);

	// tex2D
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		pipeline_layouts_.lighting,
		2,
		{ *texture_manager_.sets_.tex2d },
		{ }
	);

	// texEnv
	cmd.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics,
		pipeline_layouts_.lighting,
		3,
		{ *texture_manager_.sets_.tex_env },
		{ }
	);

	// fullscreen triangle
	cmd.draw(3, 1, 0, 0);

	cmd.endRendering();
}

void GraphicsPass::InfiniteGridPass(const vk::raii::CommandBuffer& cmd, uint32_t imageIndex, uint32_t currentFrame)
{

	vku::TransitionImageLayoutCustom(
		depth_image_, cmd,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::ImageLayout::eDepthStencilReadOnlyOptimal,
		vk::AccessFlagBits2::eShaderRead,
		vk::AccessFlagBits2::eDepthStencilAttachmentRead,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth
	);

	vk::RenderingAttachmentInfo colorAttachmentInfo{
		.imageView = swapchain_.swapchain_image_views_[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.resolveMode = {},
		.resolveImageView = {},
		.resolveImageLayout = {},
		.loadOp = vk::AttachmentLoadOp::eLoad,
		.storeOp = vk::AttachmentStoreOp::eStore,
	};
	vk::RenderingAttachmentInfo depthAtt{
		  .imageView = depth_image_view_,
		  .imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal,
		  .loadOp = vk::AttachmentLoadOp::eLoad,
		  .storeOp = vk::AttachmentStoreOp::eNone,
	};

	vk::RenderingInfo ri{
		.renderArea = { {0, 0}, swapchain_.swapchain_extent_ },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachmentInfo,
		.pDepthAttachment = &depthAtt
	};

	cmd.beginRendering(ri);

	vk::Viewport vp(
		0.0f,
		0.0f,
		static_cast<float>(swapchain_.swapchain_extent_.width),
		static_cast<float>(swapchain_.swapchain_extent_.height),
		0.0f, 1.0f
	);
	cmd.setViewport(0, vp);
	cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchain_.swapchain_extent_));

	if (infinite_pass_enable_)
	{
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines_.infinite_grid);

		uint32_t gridOffset = static_cast<uint32_t>(currentFrame * ubo_size_.grid);
		// grid set
		cmd.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics,
			pipeline_layouts_.infinite_grid,
			0,
			{ *sets_.grid },
			{ gridOffset }
		);
		cmd.draw(6, 1, 0, 0);
	}

	// ImGUI
	ImDrawData* draw_data = ImGui::GetDrawData();
	ImGui_ImplVulkan_RenderDrawData(draw_data, *cmd);

	cmd.endRendering();

	vku::TransitionImageLayoutCustom(
		depth_image_, cmd,
		vk::ImageLayout::eDepthStencilReadOnlyOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentRead,
		vk::AccessFlagBits2::eShaderRead,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eDepth
	);
}