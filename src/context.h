#pragma once

#include "swapchain.h"
#include "project_types.h"
#include "compute_pass.h"
#include "graphics_pass.h"

struct Camera;

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
	std::vector<vk::raii::CommandBuffer> command_buffers_;
	std::vector<vk::raii::CommandBuffer> compute_command_buffers_;

	vk::raii::DescriptorPool		 descriptor_pool_ = nullptr;
	vk::raii::DescriptorPool		 imgui_pool_ = nullptr;

	std::unique_ptr<ComputePass> compute_pass_;
	std::unique_ptr<GraphicsPass> graphics_pass_;
	std::unique_ptr<Swapchain> swapchain_;

	Counts counts_;

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

private:
	void DrawImgui();

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

	void CreateDescriptorPool();

	void CreateSyncObjects();

	void SetupImgui(uint32_t width, uint32_t height);

};