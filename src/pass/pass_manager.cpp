#include "context.h"
#include "swapchain.h"
#include "simulation_pass_cpu.h"
#include "simulation_pass_gpu.h"
#include "texture_manager.h"
#include "model_manager.h"
#include "graphics_pass.h"
#include "gui.h"
#include "particle_manager.h"
#include "camera.h"
#include "mouse_interactor.h"

#include "pass_manager.h"

PassManager::PassManager(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain, TextureManager& textureManager, ModelManager& modelManager)
	: context_(context), swapchain_(swapchain)
{
	CreateSyncObjects();

	particle_manager_ = std::make_unique<ParticleManager>(context, modelManager);

	if (cpu_or_gpu_ == vku::CpuOrGpu::CPU)
		sim_pass_cpu_ = std::make_unique<SimulationPassCPU>(context, *particle_manager_);
	else
		sim_pass_gpu_ = std::make_unique<SimulationPassGPU>(context, swapchain, *particle_manager_);
	graphics_pass_ = std::make_unique<GraphicsPass>(context, swapchain, textureManager, modelManager, *particle_manager_);
}

void PassManager::Update(Camera& camera, MouseInteractor& mouseInteractor, ModelManager& modelManager)
{
	if (cpu_or_gpu_ == vku::CpuOrGpu::CPU)
	{
		sim_pass_cpu_->UpdateMousePushConstant(camera, mouseInteractor, glm::vec2(swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height));

		sim_pass_cpu_->ComputeSolve(modelManager);
	}
	else if (cpu_or_gpu_ == vku::CpuOrGpu::GPU)
	{
		sim_pass_gpu_->UpdateComputeUBO(current_frame_, modelManager);

		sim_pass_gpu_->UpdateMousePushConstant(camera, mouseInteractor, glm::vec2(swapchain_.swapchain_extent_.width, swapchain_.swapchain_extent_.height));
	}

	graphics_pass_->UpdateGraphicsUBO(current_frame_, camera);
}

void PassManager::Draw(std::unique_ptr<GUI>& gui, bool paused_)
{
	auto& device = context_.device_;
	auto& queue = context_.queue_;

	const uint32_t frame = current_frame_;

	{
		device.waitForFences(*in_flight_fences_[frame], vk::True, UINT64_MAX);
		device.resetFences(*in_flight_fences_[frame]);

		if (gui->open_timestamps_)
		{
			vk::SemaphoreWaitInfo waitInfo{
				.semaphoreCount = 1,
				.pSemaphores = &*timeline_semaphore_,
				.pValues = &last_compute_timeline_
			};
			device.waitSemaphores(waitInfo, UINT64_MAX);

			sim_pass_gpu_->CalculateGpuTime();
		}
		
		//sim_pass_gpu_->ClearCpuTime();
	}

	uint32_t imageIndex = 0;
	{
		auto [result, idx] =
			swapchain_.swapchain_.acquireNextImage(
				UINT64_MAX,
				*image_available_[frame],
				nullptr
			);

		if (result == vk::Result::eErrorOutOfDateKHR) {
			RecreateSwapchain();
			return;
		}
		else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		imageIndex = idx;
	}

	// Record
	if (cpu_or_gpu_ == vku::CpuOrGpu::GPU && !paused_)
	{
		sim_pass_gpu_->RecordComputeCloth(frame, test_scene_);
		//sim_pass_gpu_->RecordComputeSoftBody(frame);

		// compute submit
		uint64_t computeSignalValue = 0;
		{
			computeSignalValue = ++timeline_value_;
			last_compute_timeline_ = computeSignalValue;

			vk::TimelineSemaphoreSubmitInfo timelineInfo{
				.waitSemaphoreValueCount = 0,
				.pWaitSemaphoreValues = nullptr,
				.signalSemaphoreValueCount = 1,
				.pSignalSemaphoreValues = &computeSignalValue
			};

			vk::SubmitInfo submitInfo{
				.pNext = &timelineInfo,
				.waitSemaphoreCount = 0,
				.pWaitSemaphores = nullptr,
				.pWaitDstStageMask = nullptr,
				.commandBufferCount = 1,
				.pCommandBuffers = &*sim_pass_gpu_->cmds_[frame],
				.signalSemaphoreCount = 1,
				.pSignalSemaphores = &*timeline_semaphore_
			};

			if (cpu_or_gpu_ == vku::CpuOrGpu::CPU)
			{
				submitInfo.pCommandBuffers = &*sim_pass_cpu_->cmds_[frame];
			}

			queue.submit(submitInfo, nullptr);
		}
	}

	graphics_pass_->RecordGraphicsCommandBuffer(imageIndex, current_frame_, cpu_or_gpu_);

	// graphics submit
	{
		vk::Semaphore waitSems[] = {
			*image_available_[frame]
		};
		vk::PipelineStageFlags waitStages[] = {
			vk::PipelineStageFlagBits::eColorAttachmentOutput
		};
		vk::Semaphore signalSems[] = {
			*image_render_finished_[imageIndex]
		};

		vk::SubmitInfo submitInfo{
			.pNext = nullptr,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = waitSems,
			.pWaitDstStageMask = waitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &*graphics_pass_->cmds_[frame],
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = signalSems
		};

		queue.submit(submitInfo, *in_flight_fences_[frame]);
	}

	{
		vk::Semaphore waitSem = *image_render_finished_[imageIndex];

		vk::PresentInfoKHR presentInfo{
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &waitSem,
			.swapchainCount = 1,
			.pSwapchains = &*swapchain_.swapchain_,
			.pImageIndices = &imageIndex
		};

		auto result = queue.presentKHR(presentInfo);
		if (result == vk::Result::eErrorOutOfDateKHR ||
			result == vk::Result::eSuboptimalKHR ||
			context_.framebuffer_resized_) {
			context_.framebuffer_resized_ = false;
			RecreateSwapchain();
		}
		else if (result != vk::Result::eSuccess) {
			throw std::runtime_error("failed to present swap chain image!");
		}
	}

	if (gui->open_timestamps_)
	{
		graphics_pass_->CalculateGpuTime();
	}

	current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void PassManager::CreateSyncObjects()
{
	in_flight_fences_.clear();

	vk::SemaphoreTypeCreateInfo semaphoreType{
		.semaphoreType = vk::SemaphoreType::eTimeline,
		.initialValue = 0
	};
	timeline_semaphore_ = vk::raii::Semaphore(context_.device_, { .pNext = &semaphoreType });
	timeline_value_ = 0;

	image_available_.clear();
	in_flight_fences_.clear();

	image_available_.reserve(MAX_FRAMES_IN_FLIGHT);
	in_flight_fences_.reserve(MAX_FRAMES_IN_FLIGHT);

	vk::SemaphoreCreateInfo binarySemaphoreInfo{}; // 기본은 binary
	vk::FenceCreateInfo fenceInfo{
		.flags = vk::FenceCreateFlagBits::eSignaled // 첫 프레임용
	};

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		image_available_.emplace_back(context_.device_, binarySemaphoreInfo);
		in_flight_fences_.emplace_back(context_.device_, fenceInfo);

		frame_timeline_done_[i] = 0;  // 해당 프레임이 마지막으로 기다린 타임라인 값
	}

	image_render_finished_.clear();
	image_render_finished_.reserve(swapchain_.image_count_);
	for (size_t i = 0; i < swapchain_.image_count_; i++)
	{

		image_render_finished_.emplace_back(context_.device_, binarySemaphoreInfo);
	}
}

void PassManager::RecreateSwapchain()
{
	swapchain_.RecreateSwapChain(context_.physical_device_, context_.device_, context_.surface_);
	graphics_pass_->CreateDepthResources();
}


PassManager::~PassManager()
{

}
