#pragma once

class Context;
class Swapchain;
class GpuSim;

class GUI
{
public:
	GUI(std::unique_ptr<Context>& ctx, GLFWwindow* glfwWindow);
	GUI(const GUI& rhs) = delete;
	GUI(GUI&& rhs) = delete;
	GUI& operator=(const GUI& rhs) = delete;
	GUI& operator=(GUI&& rhs) = delete;
	~GUI();

	vk::raii::DescriptorPool imgui_pool_{ nullptr };
	void UpdateImgui(std::unique_ptr<GpuSim>& gpuSim, vku::TestScene& testScene, std::unique_ptr<Context>& context);
};