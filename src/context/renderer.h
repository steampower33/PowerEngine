#pragma once

class Context;
class Swapchain;
class GUI;
class PassManager;
class ModelManager;
class TextureManager;
class MouseInteractor;
struct Camera;

class Renderer
{
public:
	Renderer(GLFWwindow* glfwWindow, float width, float height);
	Renderer(const Renderer& rhs) = delete;
	Renderer(Renderer&& rhs) = delete;
	Renderer& operator=(const Renderer& rhs) = delete;
	Renderer& operator=(Renderer&& rhs) = delete;
	~Renderer();

	void WaitIdle();
	void Update(Camera& camera, MouseInteractor& mouse_interactor, float dt, float& targetSimFPS, double& simDt);
	void Draw();

private:
	std::unique_ptr<Context> context_;
	std::unique_ptr<Swapchain> swapchain_;
	std::unique_ptr<GUI> gui_;
	std::unique_ptr<TextureManager> texture_manager_;
	std::unique_ptr<ModelManager> model_manager_;
	std::unique_ptr<PassManager> pass_manager_;
};