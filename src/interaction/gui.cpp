#pragma once

#include "context.h"
#include "swapchain.h"
#include "vulkan_utils.h"
#include "gpu_sim.h"
#include "graphics_context.h"
#include "texture_manager.h"

#include "gui.h"

GUI::GUI(GLFWwindow* glfwWindow, Context& context, Swapchain& swapchain)
{
	// ImGUI DescriptorPool
	std::array<vk::DescriptorPoolSize, 11> imguiPoolSizes{
		vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageTexelBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eStorageBufferDynamic, 1000},
		vk::DescriptorPoolSize{vk::DescriptorType::eInputAttachment, 1000},
	};

	vk::DescriptorPoolCreateInfo imguiPoolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 1000,
		.poolSizeCount = static_cast<uint32_t>(imguiPoolSizes.size()),
		.pPoolSizes = imguiPoolSizes.data()
	};
	imgui_pool_ = vk::raii::DescriptorPool(context.device_, imguiPoolInfo);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	io.DisplaySize.x = swapchain.swapchain_extent_.width;
	io.DisplaySize.y = swapchain.swapchain_extent_.height;

	ImGui::GetStyle().FontScaleMain = 1.5f;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	VkFormat depthFmt = static_cast<VkFormat>(vku::FindDepthFormat(context.physical_device_));
	VkFormat colorFmt = static_cast<VkFormat>(swapchain.swapchain_surface_format_.format);
	static VkFormat colorFormats[] = { colorFmt };
	ImGui_ImplGlfw_InitForVulkan(glfwWindow, true);
	ImGui_ImplVulkan_InitInfo init_info = {
		.ApiVersion = vk::ApiVersion14,
		.Instance = *context.instance_,
		.PhysicalDevice = *context.physical_device_,
		.Device = *context.device_,
		.QueueFamily = context.queue_index_,
		.Queue = *context.queue_,
		.DescriptorPool = *imgui_pool_,
		.MinImageCount = swapchain.min_image_count_,
		.ImageCount = swapchain.image_count_,
		.PipelineCache = NULL,
		.PipelineInfoMain = {
			.RenderPass = NULL,
			.Subpass = 0,
			.MSAASamples = static_cast<VkSampleCountFlagBits>(vk::SampleCountFlagBits::e1),
			.PipelineRenderingCreateInfo = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
				.pNext = NULL,
				.viewMask = 0,
				.colorAttachmentCount = 1,
				.pColorAttachmentFormats = &colorFmt,
				.depthAttachmentFormat = depthFmt,
				.stencilAttachmentFormat = VK_FORMAT_UNDEFINED
			},
		},
		.UseDynamicRendering = true,
		.Allocator = NULL,
	};

	ImGui_ImplVulkan_Init(&init_info);
}

GUI::~GUI()
{
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void GUI::Update(Context& context, GraphicsContext& graphicsContext, Swapchain& swapchain)
{
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGuiStyle& st = ImGui::GetStyle();
	st.WindowRounding = 12.f;          // 창 라운드
	st.FrameRounding = 10.f;          // 슬라이더/체크 등 라운드
	st.GrabRounding = 10.f;
	st.ScrollbarRounding = 10.f;

	st.FramePadding = ImVec2(10, 6); // 컨트롤 패딩
	st.ItemSpacing = ImVec2(10, 8); // 항목 간 간격
	st.WindowTitleAlign = ImVec2(0.5f, 0.5f);

	// 다크 테마 + 반투명 창
	ImVec4* col = st.Colors;
	col[ImGuiCol_WindowBg].w = 0.9f;          // 창 배경 알파
	col[ImGuiCol_FrameBg].w = 0.9f;           // 프레임 알파
	col[ImGuiCol_Header].w = 0.9f;           // CollapsingHeader

	ImGui::SetNextWindowBgAlpha(1.0f); // 창 자체 투명도
	ImGuiWindowFlags wf =
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoCollapse;

	auto row = [&](const char* label, auto drawControl)
		{
			ImGui::TableNextRow();

			// 1) 라벨 열
			ImGui::TableSetColumnIndex(0);
			ImGui::AlignTextToFramePadding();
			ImGui::TextUnformatted(label);

			// 2) 컨트롤 열: 위젯 폭을 컬럼의 남은 폭 전체로
			ImGui::TableSetColumnIndex(1);
			ImGui::SetNextItemWidth(-FLT_MIN);   // 해당 아이템 하나가 열 폭을 다 씀
			drawControl();
		};

	auto& labels = graphicsContext.labels_;
	auto& labelToTime = graphicsContext.label_time_;
	auto& labelToAvgTime = graphicsContext.label_avg_time_;

	ImGui::SetNextWindowSize(ImVec2(500, 0));
	ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_Once);
	if (ImGui::Begin("Stat", nullptr, wf))
	{
		is_print_timestamps = ImGui::CollapsingHeader("Solver timing", ImGuiTreeNodeFlags_DefaultOpen);
		if (is_print_timestamps)
		{
			if (ImGui::BeginTable("timing", 4))//,  ImGuiTableFlags_BordersOuter))
			{
				ImGui::TableSetupColumn("Kernel", ImGuiTableColumnFlags_WidthStretch, 0.5f);
				ImGui::TableSetupColumn("Time (ms)", ImGuiTableColumnFlags_WidthFixed, 100.0f);
				ImGui::TableSetupColumn("Avg (ms)", ImGuiTableColumnFlags_WidthFixed, 100.0f);
				ImGui::TableSetupColumn("%", ImGuiTableColumnFlags_WidthFixed, 100.0f);
				ImGui::TableHeadersRow();

				for (uint32_t i = 0; i < labels.size() - 1; i++)
				{
					DisplayKernelTiming(labels[i], labelToTime, labelToAvgTime);
				}

				DisplayKernelTiming(labels[labels.size() - 1], labelToTime, labelToAvgTime, false);

				ImGui::EndTable();
			}
			count_ = graphicsContext.time_count_;
		}

		if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::BeginTable("RenderingTable", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				int chooseTexIdx = graphicsContext.graphics_.object_ubo_data.chooseTexIdx;
				row("ChooseTexIdx", [&] { ImGui::SliderInt("##ChooseTexIdx", &chooseTexIdx, 0, graphicsContext.texture_manager_.max_texture_size); });
				graphicsContext.graphics_.object_ubo_data.chooseTexIdx = chooseTexIdx;
				ImGui::EndTable();
			}

		}
	
		if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::Checkbox("Draw Wireframe", &graphicsContext.gpu_sim_->is_wireframe_);
			ImGui::Checkbox("Draw Point", &graphicsContext.gpu_sim_->is_point_);

			ImGuiIO& io = ImGui::GetIO(); (void)io;
			ImGui::Text("Avr %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

			if (ImGui::BeginTable("SimulationTable", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{
				row("DevidingDt", [&] { ImGui::DragFloat("##DevidingDt", &graphicsContext.gpu_sim_->deviding_dt_, 1.0f, 1.0f, 240.0f); });
				row("Iterations", [&] { ImGui::DragInt("##Iterations", &graphicsContext.gpu_sim_->iterations_, 1, 1, 40); });
				row("BendCompliance", [&] { ImGui::DragFloat("##BendCompliance", &graphicsContext.gpu_sim_->bendCompliance, 0.1f, 0.0f, 1.0f); });
				row("Damping", [&] { ImGui::SliderFloat("##Damping", &graphicsContext.gpu_sim_->compute_.sim_params.damping, 0.f, 2.f, "%.3f"); });
				row("Relaxation Factor", [&] { ImGui::SliderFloat("##RelaxationFactor", &graphicsContext.gpu_sim_->compute_.sim_params.relaxationFactor, 0.f, 2.f, "%.3f"); });
				row("Max Speed", [&] { ImGui::SliderFloat("##MaxSpeed", &graphicsContext.gpu_sim_->compute_.sim_params.maxSpeed, 0.f, 500.f, "%.3f"); });
				row("CollisionMargin", [&] { ImGui::SliderFloat("##CollisionMargin", &graphicsContext.gpu_sim_->compute_.sim_params.collisionMargin, 0.0f, 1.0f, "%.3f"); });
				row("Thickness", [&] { ImGui::DragFloat("##Thickness", &graphicsContext.gpu_sim_->compute_.sim_params.thickness, 0.001f, 0.0f, 1.0f, "%.3f"); });
				row("Friction", [&] { ImGui::DragFloat("##Friction", &graphicsContext.gpu_sim_->compute_.sim_params.friction, 0.001f, 0.0f, 1.0f, "%.3f"); });

				bool windEnabled = (graphicsContext.gpu_sim_->compute_.sim_params.windTest != 0);
				row("Wind", [&] {
					if (ImGui::Checkbox("##Wind", &windEnabled)) {
						graphicsContext.gpu_sim_->compute_.sim_params.windTest = windEnabled ? 1 : 0;
					} });

					row("Wind Dir", [&] {
						ImGui::DragFloat3("##WindDir", &graphicsContext.gpu_sim_->compute_.sim_params.windDir[0], 0.1f, -1.0f, 1.0f); });

					row("Wind Strength", [&] {
						ImGui::DragFloat("##WindStrength", &graphicsContext.gpu_sim_->compute_.sim_params.windStrength, 0.1f, 0.0f, 5.0f); });

					ImGui::EndTable();
			}
		}

		if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::BeginTable("SceneTable", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{

				row("SphereCollision", [&] {
					ImGui::Checkbox("##SphereCollision", &graphicsContext.test_scene_.sphereCollision); });
				row("PinnedCorner", [&] {
					ImGui::Checkbox("##PinnedCorner", &graphicsContext.test_scene_.pinnedCorner); });
				row("TopPinnedCorner", [&] {
					ImGui::Checkbox("##TopPinnedCorner", &graphicsContext.test_scene_.topPinnedCorner); });

				ImGui::EndTable();
			}
		}

		ImGui::End();
	}

	ImGui::ShowDemoWindow();

	ImGui::Render();
}

void GUI::DisplayKernelTiming(const std::string name, std::unordered_map<std::string, double>& labelToTime, std::unordered_map<std::string, double>& labelToAvgTime, bool autoColor)
{
	double total = labelToAvgTime["Total"];
	float percentage = (total > 0) ? float(labelToAvgTime[name] / total * 100) : 0.0f;

	bool shouldPop = false;

	if (autoColor)
	{
		if (percentage > 10 || percentage == 0.0f)
		{
			ImVec4 textColor = color_low;
			if (percentage > 30)
				textColor = color_high;
			else if (percentage > 10)
				textColor = color_mid;
			else if (percentage == 0.0f)
				textColor = color_disabled;
			ImGui::PushStyleColor(ImGuiCol_Text, textColor);
			shouldPop = true;
		}
	}

	auto RightText = [&](const char* text)
		{
			float w = ImGui::GetColumnWidth();
			float tw = ImGui::CalcTextSize(text).x;
			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (w - tw));
			ImGui::TextUnformatted(text);
		};

	ImGui::TableNextColumn();
	ImGui::Text(name.c_str());
	ImGui::TableNextColumn();
	RightText(std::format("{:.3f}", labelToTime[name]).c_str());
	ImGui::TableNextColumn();
	RightText(std::format("{:.3f}", labelToAvgTime[name] / count_).c_str());
	ImGui::TableNextColumn();
	RightText(std::format("{:.3f}", percentage).c_str());

	if (shouldPop)
	{
		ImGui::PopStyleColor();
	}
}