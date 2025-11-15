#include "context.h"
#include "swapchain.h"
#include "gui.h"
#include "graphics_context.h"
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
	model_manager_ = std::make_unique<ModelManager>(*context_, *graphics_context_);
	texture_manager_ = std::make_unique<TextureManager>(*context_, *graphics_context_);
	graphics_context_ = std::make_unique<GraphicsContext>(glfwWindow, *context_, *swapchain_, *texture_manager_, *model_manager_);
}

Renderer::~Renderer()
{

}

void Renderer::WaitIdle()
{
	context_->WaitIdle();
}

void Renderer::Update(Camera& camera, MouseInteractor& mouse_interactor, float dt)
{
	mouse_interactor.Update(camera, glm::vec2(swapchain_->swapchain_extent_.width, swapchain_->swapchain_extent_.height), model_manager_->models);

	gui_->Update(*context_, *graphics_context_, *swapchain_);
	graphics_context_->Update(camera);
}

void Renderer::Draw()
{
	graphics_context_->Draw(gui_);
}

