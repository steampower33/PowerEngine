#pragma once

class Context;
class Swapchain;
class GraphicsContext;
class GpuSim;

class GUI
{
public:
	GUI(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain);
	GUI(const GUI& rhs) = delete;
	GUI(GUI&& rhs) = delete;
	GUI& operator=(const GUI& rhs) = delete;
	GUI& operator=(GUI&& rhs) = delete;
	~GUI();

	vk::raii::DescriptorPool imgui_pool_{ nullptr };
	uint32_t count_ = 0;

	ImVec4 color_high = ImVec4(1.000f, 0.244f, 0.000f, 1.000f);
	ImVec4 color_mid = ImVec4(1.000f, 0.602f, 0.000f, 1.000f);
	ImVec4 color_low = ImVec4(1.000f, 0.889f, 0.000f, 1.000f);
	ImVec4 color_disabled = ImVec4(0.5f, 0.5f, 0.5f, 1.000f);

	bool is_print_timestamps = false;

	void SetStyle();
	void Update(Context& context, GraphicsContext& graphicsContext, Swapchain& swapchain);
	void DisplayKernelTiming(const std::string name, std::unordered_map<std::string, double>& labelToTime, std::unordered_map<std::string, double>& labelToAvgTime, bool autoColor = true);

	template<typename RowFn, typename UBOData>
	void SetLightGUI(RowFn&& row, UBOData& data);
	template<typename RowFn, typename UBOData>
	void SetObjectGUI(RowFn&& row, UBOData& data);
	template<typename RowFn>
	void SetSolverTimeingGUI(RowFn&& row, GraphicsContext& graphicsContext);
	template<typename RowFn>
	void SetSimulationGUI(RowFn&& row, GraphicsContext& graphicsContext);
	template<typename RowFn, typename Scene>
	void SetTestSceneGUI(RowFn&& row, Scene& scene);
};