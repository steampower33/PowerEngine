#include "context.h"
#include "swapchain.h"
#include "gui.h"
#include "pass_manager.h"
#include "model_manager.h"
#include "texture_manager.h"
#include "mouse_interactor.h"
#include "camera.h"

#include "renderer.h"

Renderer::Renderer(GLFWwindow* glfwWindow, float width, float height)
{
	context_ = std::make_unique<Context>(glfwWindow, width, height);
	swapchain_ = std::make_unique<Swapchain>(glfwWindow, *context_);
	gui_ = std::make_unique<GUI>(glfwWindow, *context_, *swapchain_);
	texture_manager_ = std::make_unique<TextureManager>(*context_);
	model_manager_ = std::make_unique<ModelManager>(*context_, *pass_manager_, *texture_manager_);
	pass_manager_ = std::make_unique<PassManager>(glfwWindow, *context_, *swapchain_, *texture_manager_, *model_manager_);
}

Renderer::~Renderer()
{

}

void Renderer::WaitIdle()
{
	context_->WaitIdle();
}

void Renderer::Update(Camera& camera, MouseInteractor& mouse_interactor, float dt, float& targetSimFPS, double& simDt)
{
	gui_->Update(*context_, *pass_manager_, *swapchain_, targetSimFPS, simDt, *model_manager_, *texture_manager_, camera);
	pass_manager_->Update(camera, mouse_interactor, *model_manager_);

	mouse_interactor.Update(camera, glm::vec2(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height), model_manager_->models_);
}

void Renderer::Draw()
{
	pass_manager_->Draw(gui_);
}

