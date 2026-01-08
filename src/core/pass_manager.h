#pragma once

#include "vulkan_utils.h"

class Context;
class Swapchain;
class SimPassGPU;
class TextureManager;
class ModelManager;
class GraphicsPass;
class GUI;
class ParticleManager;
class Camera;
class MouseInteractor;

class PassManager
{
public:
	PassManager(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain, TextureManager& textureManager, ModelManager& modelManager);
	PassManager(const PassManager& rhs) = delete;
	PassManager(PassManager&& rhs) = delete;
	PassManager& operator=(const PassManager& rhs) = delete;
	PassManager& operator=(PassManager&& rhs) = delete;
	~PassManager();

	void Update(Camera& camera, MouseInteractor& mouseInteractor, ModelManager& modelManager, bool paused);
	void Draw(std::unique_ptr<GUI>& gui, bool paused);

	vku::TestScene test_scene_;

	std::unique_ptr<ParticleManager> particle_manager_;
	std::unique_ptr<SimPassGPU> sim_pass_gpu_;
	std::unique_ptr<GraphicsPass> graphics_pass_;

	vk::raii::Semaphore timeline_semaphore_{ nullptr };
	uint64_t timeline_value_{ 0 };
	uint64_t last_compute_timeline_{ 0 };

	std::vector<vk::raii::Fence> in_flight_fences_;
	std::vector<vk::raii::Semaphore> image_available_;
	std::vector<vk::raii::Semaphore> image_render_finished_;

	std::array<uint64_t, MAX_FRAMES_IN_FLIGHT> frame_timeline_done_{};

	uint32_t current_frame_{ 0 };

private:
	Context& context_;
	Swapchain& swapchain_;

private:
	void CreateSyncObjects();
	void RecreateSwapchain();

};