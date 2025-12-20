#pragma once

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

	GLFWwindow* glfw_window_;

	vk::raii::Instance               instance_{ nullptr };
	vk::raii::DebugUtilsMessengerEXT debug_messenger_{ nullptr };
	vk::raii::SurfaceKHR             surface_{ nullptr };
	vk::raii::PhysicalDevice         physical_device_{ nullptr };
	vk::raii::Device                 device_{ nullptr };
	uint32_t                         queue_index_ = ~0;
	vk::raii::Queue                  queue_{ nullptr };
	vk::raii::CommandPool			 command_pool_{ nullptr };

	bool framebuffer_resized_{ false };

	std::vector<const char*> required_device_extension_ = {
		vk::KHRSwapchainExtensionName,
		vk::KHRSpirv14ExtensionName,
		vk::KHRSynchronization2ExtensionName,
		vk::KHRCreateRenderpass2ExtensionName,
		vk::EXTShaderAtomicFloatExtensionName
	};

	void WaitIdle();

#ifdef NDEBUG
	bool enableValidationLayers = false;
#else
	bool enableValidationLayers = true;
#endif

	enum DebugMode {
		NORMAL,
		FOCUS
	};
	DebugMode debug_mode_ = DebugMode::FOCUS;

	static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback_ErrorOnly(
		vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
		vk::DebugUtilsMessageTypeFlagsEXT type,
		const vk::DebugUtilsMessengerCallbackDataEXT* cb,
		void* userData
	);
	static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback_Focus(
		vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
		vk::DebugUtilsMessageTypeFlagsEXT type,
		const vk::DebugUtilsMessengerCallbackDataEXT* cb,
		void* userData
	);

private:
	void CreateInstance();
	std::vector<const char*> GetRequiredExtensions();


	void CreateSurface();
	void PickPhysicalDevice();
	void CreateLogicalDevice();
	void CreateCommandPool();


};