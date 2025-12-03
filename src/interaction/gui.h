#pragma once

class Context;
class Swapchain;
class PassManager;
class SimulationPassGPU;
class SimulationPassCPU;
class Model;
class ModelManager;
class TextureManager;
class GraphicsPass;
class ParticleManager;

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
	void Update(Context& context, PassManager& passManager, Swapchain& swapchain, float& targetSimFPS, double& simDt, ModelManager& modelManager, TextureManager& textureManager);
	void DisplayKernelTiming(const std::string name, std::unordered_map<std::string, double>& labelToTime, std::unordered_map<std::string, double>& labelToAvgTime, bool autoColor = true);

	template<typename RowFn>
	void SetRenderingGUI(RowFn&& row, GraphicsPass& graphicsPass, TextureManager& textureManager);
	template<typename RowFn, typename Objects, typename ClothUBO>
	void SetObjectsGUI(RowFn&& row, Objects& objects, ClothUBO& clothUBO);
	template<typename RowFn>
	void SetTimeingGUI(RowFn&& row, SimulationPassGPU& sim, GraphicsPass& graphicsPass);
	template<typename RowFn, typename Sim>
	void SetSimulationGUI(RowFn&& row, PassManager& passManager, Sim& sim, float& targetSimFPS, double& simDt);
	template<typename RowFn, typename Scene>
	void SetTestSceneGUI(RowFn&& row, Scene& scene);
	template<typename RowFn>
	void SetStatGUI(RowFn&& row, PassManager& passManager);
};