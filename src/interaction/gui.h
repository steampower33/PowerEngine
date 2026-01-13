#pragma once

class Context;
class Swapchain;
class PassManager;
class ClothSimPass;
class SimulationPassCPU;
class Model;
class ModelManager;
class TextureManager;
class GraphicsPass;
class ClothParticleManager;
class Camera;

class GUI
{
public:
	GUI(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain, TextureManager& textureManager, ModelManager& modelManager, PassManager& passManager);
	GUI(const GUI& rhs) = delete;
	GUI(GUI&& rhs) = delete;
	GUI& operator=(const GUI& rhs) = delete;
	GUI& operator=(GUI&& rhs) = delete;
	~GUI();

	Context& context_;
	Swapchain& swapchain_;
	TextureManager& texture_manager_;
	ModelManager& model_manager_;
	PassManager& pass_manager_;

	float font_scale = 0.0f;

	vk::raii::DescriptorPool imgui_pool_{ nullptr };
	uint32_t count_ = 0;

	ImVec4 color_high = ImVec4(1.000f, 0.244f, 0.000f, 1.000f);
	ImVec4 color_mid = ImVec4(1.000f, 0.602f, 0.000f, 1.000f);
	ImVec4 color_low = ImVec4(1.000f, 0.889f, 0.000f, 1.000f);
	ImVec4 color_disabled = ImVec4(0.5f, 0.5f, 0.5f, 1.000f);

	bool open_timestamps_ = false;

	std::vector<std::pair<std::string, VkDescriptorSet>> imgui_id_;

	void SetStyle();
	void Update(float& targetSimFPS, double& simDt, Camera& camera, bool& paused, bool& pauseEachframe);
	void DisplayKernelTiming(const std::string name, std::unordered_map<std::string, double>& labelToTime, std::unordered_map<std::string, double>& labelToAvgTime, bool autoColor = true);

	template<typename RowFn>
	void SetRenderingGUI(RowFn&& row);
	template<typename RowFn>
	void SetObjectsGUI(RowFn&& row);
	template<typename RowFn>
	void SetTimeingGUI(RowFn&& row);
	template<typename RowFn>
	void SetClothGUI(RowFn&& row, float& targetSimFPS, double& simDt, bool& paused, bool& pauseEachframe);
	template<typename RowFn>
	void SetStatGUI(RowFn&& row);
	template<typename RowFn>
	void SetCameraGUI(RowFn&& row, Camera& camera);
};