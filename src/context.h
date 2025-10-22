#pragma once

class Swapchain;
struct Vertex;
struct Camera;
class Model;
class Texture2D;

class Context
{
public:
	Context(GLFWwindow* glfwWindow, uint32_t width, uint32_t height);
	Context(const Context& rhs) = delete;
	Context(Context&& rhs) = delete;
	Context& operator=(const Context& rhs) = delete;
	Context& operator=(Context&& rhs) = delete;
	~Context();

	void Draw(Camera& camera, float dt);
	void WaitIdle();

private:
	GLFWwindow* glfw_window_;

	vk::raii::Context                context_;
	vk::raii::Instance               instance_ = nullptr;
	vk::raii::DebugUtilsMessengerEXT debug_messenger_ = nullptr;
	vk::raii::SurfaceKHR             surface_ = nullptr;
	vk::raii::PhysicalDevice         physical_device_ = nullptr;
	vk::SampleCountFlagBits			 msaa_samples_ = vk::SampleCountFlagBits::e1;
	vk::raii::Device                 device_ = nullptr;

	uint32_t                         queue_index_ = ~0;
	vk::raii::Queue                  queue_ = nullptr;

	vk::raii::CommandPool			 command_pool_ = nullptr;

	vk::raii::DescriptorPool		 descriptor_pool_ = nullptr;
	vk::raii::DescriptorPool		 imgui_pool_ = nullptr;

	std::unique_ptr<Swapchain>       swapchain_;

	// Resources for the graphics part of the example
	struct Graphics {
		struct UniformData {
			glm::mat4 model;
			glm::mat4 view;
			glm::mat4 proj;
		} uniformData;
		std::vector<vk::raii::Buffer> uniform_buffers_;
		std::vector<vk::raii::DeviceMemory> uniform_buffers_memory_;
		std::vector<void*> uniform_buffers_mapped_;

		vk::raii::DescriptorSetLayout descriptor_set_layout_ = nullptr;
		std::vector<vk::raii::DescriptorSet> descriptor_sets_;

		vk::raii::PipelineLayout pipeline_layout_ = nullptr;
		vk::raii::Pipeline pipeline_ = nullptr;

		std::vector<vk::raii::CommandBuffer> command_buffers_;
	} graphics_;

	//struct Compute {
	//	struct UniformData {
	//		float dt{ 0.0f };
	//	} uniformData;

	//	std::vector<vk::raii::Buffer> uniform_buffers_;
	//	std::vector<vk::raii::DeviceMemory> uniform_buffers_memory_;
	//	std::vector<void*> uniform_buffers_mapped_;

	//	std::vector<vk::raii::Buffer> shader_storage_buffers_;
	//	std::vector<vk::raii::DeviceMemory> shader_storage_buffers_memory_;

	//	vk::raii::DescriptorSetLayout descriptor_set_layout_ = nullptr;
	//	std::vector<vk::raii::DescriptorSet> descriptor_sets_;

	//	vk::raii::PipelineLayout pipeline_layout_ = nullptr;
	//	vk::raii::Pipeline pipeline_ = nullptr;

	//	std::vector<vk::raii::CommandBuffer> command_buffers_;
	//} compute_;

	struct Counts {
		uint32_t ubo = 0;
		uint32_t sb = 0;
		uint32_t sampler = 0;
		uint32_t layout = 0;
	} counts_;

	vk::raii::Semaphore semaphore_ = nullptr;
	uint64_t timeline_value_ = 0;
	std::vector<vk::raii::Fence> in_flight_fences_;
	uint32_t current_frame_ = 0;

	bool framebuffer_resized_ = false;

	std::vector<const char*> required_device_extension_ = {
		vk::KHRSwapchainExtensionName,
		vk::KHRSpirv14ExtensionName,
		vk::KHRSynchronization2ExtensionName,
		vk::KHRCreateRenderpass2ExtensionName
	};

	uint32_t particle_count = 256;
	std::unique_ptr<Model> sphere_;
	std::unique_ptr<Texture2D> texture_;

private:
	void DrawImgui();
	void UpdateUniformBuffer(Camera& camera);
	void Transition_image_layout(
		vk::Image& image,
		const vk::raii::CommandBuffer& cmd,
		vk::ImageLayout old_layout,
		vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask,
		vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask
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
	void CreateUniformBuffers();
	void CreateDescriptorPools();
	void CreateDescriptorSets();
	void CreateGraphicsPipelines();
	void CreateSyncObjects();

	void SetupImgui(uint32_t width, uint32_t height);

};