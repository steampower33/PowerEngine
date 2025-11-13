#include "context.h"
#include "swapchain.h"
#include "vulkan_utils.h"
#include "gpu_sim.h"

#include "gui.h"

GUI::GUI(std::unique_ptr<Context>& ctx, GLFWwindow* glfwWindow)
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
	imgui_pool_ = vk::raii::DescriptorPool(ctx->device_, imguiPoolInfo);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	io.DisplaySize.x = ctx->swapchain_->swapchain_extent_.width;
	io.DisplaySize.y = ctx->swapchain_->swapchain_extent_.height;

	ImGui::GetStyle().FontScaleMain = 1.5f;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	VkFormat depthFmt = static_cast<VkFormat>(vku::FindDepthFormat(ctx->physical_device_));
	VkFormat colorFmt = static_cast<VkFormat>(ctx->swapchain_->swapchain_surface_format_.format);
	static VkFormat colorFormats[] = { colorFmt };
	ImGui_ImplGlfw_InitForVulkan(glfwWindow, true);
	ImGui_ImplVulkan_InitInfo init_info = {
		.ApiVersion = vk::ApiVersion14,
		.Instance = *ctx->instance_,
		.PhysicalDevice = *ctx->physical_device_,
		.Device = *ctx->device_,
		.QueueFamily = ctx->queue_index_,
		.Queue = *ctx->queue_,
		.DescriptorPool = *imgui_pool_,
		.MinImageCount = ctx->swapchain_->min_image_count_,
		.ImageCount = ctx->swapchain_->image_count_,
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

void GUI::UpdateImgui(std::unique_ptr<GpuSim>& gpuSim, vku::TestScene& testScene)
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
	col[ImGuiCol_WindowBg].w = 0.1f;          // 창 배경 알파
	col[ImGuiCol_FrameBg].w = 0.7f;           // 프레임 알파
	col[ImGuiCol_Header].w = 0.5f;           // CollapsingHeader

	ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_Once);
	ImGui::SetNextWindowBgAlpha(0.85f); // 창 자체 투명도
	ImGuiWindowFlags wf = ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_AlwaysAutoResize |
		ImGuiWindowFlags_NoCollapse;

	if (ImGui::Begin("Options", nullptr, wf))
	{
		ImGui::Checkbox("Draw Wireframe", &gpuSim->is_wireframe_);

		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui::Text("Avr %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

		auto row = [&](const char* label, auto drawControl)
			{
				ImGui::TableNextRow();

				// 1) 라벨 열
				ImGui::TableSetColumnIndex(0);
				ImGui::AlignTextToFramePadding();
				ImGui::TextUnformatted(label);

				// 2) 컨트롤 열: 위젯 폭을 컬럼의 남은 폭 전체로
				ImGui::TableSetColumnIndex(1);
				ImGui::SetNextItemWidth(-FLT_MIN);   // ★ 핵심: 해당 아이템 하나가 열 폭을 다 씀
				drawControl();
			};

		if (ImGui::BeginTable("sim_tbl", 2,
			ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
		{
			row("Iterations", [&] { ImGui::DragInt("##Iterations", &gpuSim->iterations_, 1, 1, 50); });
			row("Damping", [&] { ImGui::SliderFloat("##Damping", &gpuSim->compute_.sim_params.damping, 0.f, 2.f, "%.3f"); });
			row("Relaxation Factor", [&] { ImGui::SliderFloat("##RelaxationFactor", &gpuSim->compute_.sim_params.relaxationFactor, 0.f, 2.f, "%.3f"); });
			row("Max Speed", [&] { ImGui::SliderFloat("##MaxSpeed", &gpuSim->compute_.sim_params.maxSpeed, 0.f, 500.f, "%.3f"); });

			bool windEnabled = (gpuSim->compute_.sim_params.windTest != 0);
			row("Wind", [&] {
				if (ImGui::Checkbox("##Wind", &windEnabled)) {
					gpuSim->compute_.sim_params.windTest = windEnabled ? 1 : 0;
				} });

				row("Wind Dir", [&] {
					ImGui::DragFloat3("##WindDir", &gpuSim->compute_.sim_params.windDir[0], 0.1f, -1.0f, 1.0f); });

				row("Wind Strength", [&] {
					ImGui::DragFloat("##WindStrength", &gpuSim->compute_.sim_params.windStrength, 0.1f, 0.0f, 5.0f); });

				ImGui::EndTable();
		}

		if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
			if (ImGui::BeginTable("scene_tbl", 2,
				ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV))
			{

				row("Sphere Collision", [&] {
					ImGui::Checkbox("##SphereCollision", &testScene.sphereCollision); });
				row("Pinned Corner", [&] {
					ImGui::Checkbox("##PinnedCorner", &testScene.pinnedCorner); });

				ImGui::EndTable();
			}
		}
	}
	ImGui::End();


	//ImGui::ShowDemoWindow();

	ImGui::Render();
}