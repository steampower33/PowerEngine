#pragma once

class Swapchain;
struct Vertex;
struct Camera;
class Model;
class Texture2D;
class MouseInteractor;
class CpuSim;
class GpuSim;

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
	void Draw(bool& printTimestamp);
	void WaitIdle();

	uint32_t timestampSteps = 0;

	GLFWwindow* glfw_window_;

	vk::raii::Instance               instance_{ nullptr };
	vk::raii::DebugUtilsMessengerEXT debug_messenger_{ nullptr };
	vk::raii::SurfaceKHR             surface_{ nullptr };
	vk::raii::PhysicalDevice         physical_device_{ nullptr };
	vk::SampleCountFlagBits			 msaa_samples_ = vk::SampleCountFlagBits::e1;
	bool use_float_atomics_{ false };
	vk::raii::Device                 device_{ nullptr };

	uint32_t                         queue_index_ = ~0;
	vk::raii::Queue                  queue_{ nullptr };

	vk::raii::CommandPool			 command_pool_{ nullptr };

	vk::raii::DescriptorPool		 descriptor_pool_{ nullptr };

	std::unique_ptr<Swapchain>       swapchain_{ nullptr };

	vk::raii::QueryPool				 timestamp_pool_{ nullptr };

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
		vk::KHRCreateRenderpass2ExtensionName,
		vk::EXTShaderAtomicFloatExtensionName
	};

	enum CpuOrGpu {
		CPU,
		GPU
	};

	CpuOrGpu cpu_or_gpu_ = CpuOrGpu::GPU;
	vku::TestScene test_scene_;

	std::unique_ptr<CpuSim> cpu_sim_;
	std::unique_ptr<GpuSim> gpu_sim_;

	// |===== Particle Info =====|
	const uint32_t Nx_ = 32;
	const uint32_t Ny_ = 32;
	const float spacing_ = 0.2;

	uint32_t particles_size_ = Nx_ * Ny_;
	uint32_t indices_size_ = 0;
	uint32_t edges_size_ = 0;

	// |===== Push Constant =====|
	struct ClothPC {
		uint32_t Nx;
		uint32_t Ny;
	} cloth_pc_;
	static_assert(sizeof(ClothPC) % 4 == 0, "push constant must be multiple of 4 bytes");

	vku::Counts counts_;
	uint32_t sim_count = 0;

	struct CommandBuffers {
		std::vector<vk::raii::CommandBuffer> compute;
		std::vector<vk::raii::CommandBuffer> graphics;
	} cmds_;

	// |===== Graphics Info =====|
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
			vk::raii::Pipeline model{ nullptr };
		} pipelines;

	} graphics_;

	// |===== Model & Texture =====|
	static constexpr uint32_t kMaxObjects = 8;
	uint32_t model_count_ = 0;
	std::vector<std::unique_ptr<Model>> models;
	std::unique_ptr<Texture2D> texture_{ nullptr };

	// |===== Depth Image =====|
	vk::raii::Image depth_image_ = nullptr;
	vk::raii::DeviceMemory depth_image_memory_ = nullptr;
	vk::raii::ImageView depth_image_view_ = nullptr;

private:

	void UpdateMouseInteractor(Camera& camera, MouseInteractor& mouse_interactor);
	void UpdatePushContants();
	void UpdateGraphicsUBO(Camera& camera);

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
	void CreateQueryPool();

	void CreateDescriptorSetLayout();
	void CreateDescriptorPools();

	void CreateUniformBuffers();

	void CreateDescriptorSets();
	void CreateGraphicsPipelines();
	void CreateSyncObjects();

	void CreateDepthResources();
};