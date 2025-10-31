#pragma once

class Swapchain;
struct Vertex;
struct Camera;
class Model;
class Texture2D;
class MouseInteractor;

#include "vulkan_utils.h"

class Context
{
public:
	Context(GLFWwindow* glfwWindow, uint32_t width, uint32_t height);
	Context(const Context& rhs) = delete;
	Context(Context&& rhs) = delete;
	Context& operator=(const Context& rhs) = delete;
	Context& operator=(Context&& rhs) = delete;
	~Context();

	void Update(Camera& camera, MouseInteractor& mouse_interactor, float dt);
	void Draw();
	void WaitIdle();

private:
	GLFWwindow* glfw_window_;

	vk::raii::Instance               instance_{ nullptr };
	vk::raii::DebugUtilsMessengerEXT debug_messenger_{ nullptr };
	vk::raii::SurfaceKHR             surface_{ nullptr };
	vk::raii::PhysicalDevice         physical_device_{ nullptr };
	vk::SampleCountFlagBits			 msaa_samples_ = vk::SampleCountFlagBits::e1;
	vk::raii::Device                 device_{ nullptr };

	uint32_t                         queue_index_ = ~0;
	vk::raii::Queue                  queue_{ nullptr };

	vk::raii::CommandPool			 command_pool_{ nullptr };

	vk::raii::DescriptorPool		 descriptor_pool_{ nullptr };
	vk::raii::DescriptorPool		 imgui_pool_{ nullptr };

	std::unique_ptr<Swapchain>       swapchain_{ nullptr };

	vk::raii::Semaphore semaphore_{ nullptr };
	uint64_t timeline_value_{ 0 };
	std::vector<vk::raii::Fence> in_flight_fences_;
	uint32_t current_frame_{ 0 };
	uint32_t read_set_{ 0 };

	bool framebuffer_resized_{ false };

	std::vector<const char*> required_device_extension_ = {
		vk::KHRSwapchainExtensionName,
		vk::KHRSpirv14ExtensionName,
		vk::KHRSynchronization2ExtensionName,
		vk::KHRCreateRenderpass2ExtensionName
	};

private:
	vku::Counts counts_;

	struct Compute {
		struct SimParams {
			float dt;
			float gravityY;
			uint32_t numIters;
			float stretchCompliance; // α_stretch
			float restitution;       // 충돌 반발
		} sim_params;
		vk::raii::Buffer sim_params_ubo{ nullptr };
		vk::raii::DeviceMemory sim_params_ubo_memory{ nullptr };
		void* sim_params_ubo_mapped{ nullptr };

		struct Particle {
			glm::vec4 x;      // xyz=pos, w=invMass
			glm::vec4 v;      // xyz=vel
			glm::vec4 xp;     // xyz=predicted pos (x + dt*v), w=reserved
		};

		struct Edge {
			uint32_t i, j;
			float restLen;
			float compliance; // per-constraint alpha override (0이면 SimParams 사용)
			float lambda;     // XPBD 누적 라그랑주 승수
		};

		vk::raii::DescriptorSetLayout descriptor_set_layout{ nullptr };
		vk::raii::DescriptorSet descriptor_set{ nullptr };

		struct PipelineLayouts {
			vk::raii::PipelineLayout integrate{ nullptr };
		} pipeline_layouts;

		struct Pipelines {
			vk::raii::Pipeline integrate{ nullptr };
		} pipelines;
	} compute_;

	struct Graphics {
		struct GlobalUboData {
			glm::mat4 view;
			glm::mat4 proj;
		} global_ubo_data;
		vk::raii::Buffer global_ubo{ nullptr };
		vk::raii::DeviceMemory global_ubo_memory{ nullptr };
		void* global_ubo_mapped{ nullptr };
		vk::DeviceSize global_slot_size;

		struct ObjectUboData {
			glm::mat4 model;
		} object_ubo_data;
		vk::raii::Buffer object_ubo{ nullptr };
		vk::raii::DeviceMemory object_ubo_memory{ nullptr };
		void* object_ubo_mapped{ nullptr };
		vk::DeviceSize object_slot_size;

		vk::raii::DescriptorSetLayout global_set_layout{ nullptr };
		vk::raii::DescriptorSet global_set{ nullptr };
		vk::raii::DescriptorSetLayout object_set_layout{ nullptr };
		vk::raii::DescriptorSet object_set{ nullptr };

		struct PipelineLayouts {
			vk::raii::PipelineLayout model{ nullptr };
		} pipeline_layouts;

		struct Pipelines {
			vk::raii::Pipeline sphere{ nullptr };
		} pipelines;

		std::vector<vk::raii::CommandBuffer> command_buffers;
	} graphics_;

	static constexpr uint32_t kMaxObjects = 8;
	uint32_t model_count_ = 0;
	std::vector<std::unique_ptr<Model>> models;

	std::unique_ptr<Texture2D> texture_{ nullptr };
private:
	vk::raii::Image depth_image_ = nullptr;
	vk::raii::DeviceMemory depth_image_memory_ = nullptr;
	vk::raii::ImageView depth_image_view_ = nullptr;

private:
	void DrawImgui();

	void UpdateMouseInteractor(Camera& camera, MouseInteractor& mouse_interactor);
	void UpdateComputeUBO();
	void UpdateGraphicsUBO(Camera& camera);

	void AddComputeToComputeBarrier(const vk::raii::CommandBuffer& cmd, vk::Buffer buffer);
	void AddGraphicsToComputeBarrier(const vk::raii::CommandBuffer& cmd, vk::Buffer buffer);
	void AddComputeToGraphicsBarrier(const vk::raii::CommandBuffer& cmd, vk::Buffer buffer);

	void RecordGraphicsCommandBuffer(uint32_t imageIndex);
	void TransitionImageLayout(
		vk::Image& image,
		const vk::raii::CommandBuffer& cmd,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask
	);
	void TransitionImageLayoutCustom(
		vk::raii::Image& image,
		const vk::raii::CommandBuffer& cmd,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask,
		vk::ImageAspectFlags aspect_mask
	);

private:
	void CreateInstance();
	std::vector<const char*> GetRequiredExtensions();
	void SetupDebugMessenger();
	static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*);
	void CreateSurface();
	void PickPhysicalDevice();
	vk::SampleCountFlagBits GetMaxUsableSampleCount();
	void CreateLogicalDevice();
	void CreateCommandPool();
	void CreateCommandBuffers();

	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();

	void CreateUniformBuffers();

	void CreateDescriptorSets();
	void CreateComputePipelines();
	void CreateGraphicsPipelines();
	void CreateSyncObjects();

	void CreateDepthResources();

	void SetupImgui(uint32_t width, uint32_t height);

};