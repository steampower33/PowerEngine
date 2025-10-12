#pragma once

#include "preamble.hpp"
#include "window.hpp"

class Swapchain;

class Context
{
public:
	Context(GLFWwindow* glfwWindow);
	Context(const Context& rhs) = delete;
	Context(Context&& rhs) = delete;
	~Context();

	Context& operator=(const Context& rhs) = delete;
	Context& operator=(Context&& rhs) = delete;

	void draw();

	vk::raii::Context                context_;
	vk::raii::Instance               instance_ = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger_ = nullptr;
	vk::raii::SurfaceKHR             surface_ = nullptr;
	vk::raii::PhysicalDevice         physicalDevice_ = nullptr;
	vk::raii::Device                 device_ = nullptr;
	uint32_t                         queueIndex_ = ~0;
	vk::raii::Queue                  queue_ = nullptr;
	vk::raii::CommandPool			 commandPool_ = nullptr;
	vk::raii::DescriptorSetLayout	 descriptorSetLayout_ = nullptr;
	vk::raii::PipelineLayout		 pipelineLayout_ = nullptr;
	vk::raii::Pipeline				 graphicsPipeline_ = nullptr;

	vk::raii::DescriptorPool		 descriptorPool_ = nullptr;

	bool framebufferResized_ = false;

	std::vector<const char*> requiredDeviceExtension_ = {
		vk::KHRSwapchainExtensionName,
		vk::KHRSpirv14ExtensionName,
		vk::KHRSynchronization2ExtensionName,
		vk::KHRCreateRenderpass2ExtensionName
	};

	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;

private:
	GLFWwindow* glfwWindow_;
	Swapchain* swapchain_;

	void createInstance();
	std::vector<const char*> getRequiredExtensions();
	void setupDebugMessenger();
	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*);
	void createSurface();
	void pickPhysicalDevice();
	void createLogicalDevice();
	void createCommandPool();
	
	void createDescriptorSetLayout();
	void createGraphicsPipeline();

	void createDescriptorPool();

	static std::vector<char> readFile(const std::string& filename);
};