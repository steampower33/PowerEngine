#include "context.h"
#include "camera.h"
#include "mouse_interactor.h"

#include "window.h"

Window::Window()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	glfw_window_ = glfwCreateWindow(init_width_, init_height_, "Vulkan", nullptr, nullptr);

	// 이 윈도우에 this를 매달아둔다
	glfwSetWindowUserPointer(glfw_window_, this);

	// 콜백 등록: static 트램펄린
	glfwSetFramebufferSizeCallback(glfw_window_, &Window::FramebufferResizeCallback);
	glfwSetCursorPosCallback(glfw_window_, &Window::CursorPosCallback);
	glfwSetKeyCallback(glfw_window_, &Window::KeyCallback);
	glfwSetMouseButtonCallback(glfw_window_, &Window::MouseButtonCallback);

	if (!glfw_window_)
	{
		std::cerr << "Failure creating glfw window " << std::endl;
	}

	ctx_ = std::make_unique<Context>(glfw_window_, init_width_, init_height_);
	camera_ = std::make_unique<Camera>();
	mouse_interactor_ = std::make_unique<MouseInteractor>();

}

// --- static 트램펄린들 ---
void Window::FramebufferResizeCallback(GLFWwindow* w, int width, int height) {
	if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w))) {
		self->OnFramebufferResize(width, height);
	}
}

void Window::CursorPosCallback(GLFWwindow* w, double x, double y) {
	if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w))) {
		self->OnCursorPos(x, y);
	}
}

void Window::KeyCallback(GLFWwindow* w, int key, int scancode, int action, int mods) {
	if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w))) {
		self->OnKey(key, scancode, action, mods);
	}
}

void Window::MouseButtonCallback(GLFWwindow* w, int button, int action, int mods)
{
	if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w))) {
		self->OnMouseClick(button, action, mods);
	}
}

// --- 실제 인스턴스 핸들러들 ---
void Window::OnFramebufferResize(int, int)
{
	framebuffer_resized_ = true;
}

void Window::OnCursorPos(double xpos, double ypos)
{
	static double lastX = xpos;
	static double lastY = ypos;

	if (mouse_enabled_) {
		// 카메라 모드로 처음 진입한 경우, 이전 위치를 현재 위치로 초기화하고 끝낸다.
		// 이렇게 하면 카메라가 튀는 현상을 방지할 수 있다.
		if (first_mouse_) {
			lastX = xpos;
			lastY = ypos;
			first_mouse_ = false;
		}

		// 이전 프레임과의 차이(offset)를 계산한다.
		double xoffset = xpos - lastX;
		double yoffset = lastY - ypos; // Y는 상하 반전 (y값이 작아질수록 위로 감)

		// 카메라의 각도를 업데이트한다.
		camera_->yaw += static_cast<float>(xoffset) * camera_->sensitivity;
		camera_->pitch += static_cast<float>(yoffset) * camera_->sensitivity;
		camera_->pitch = glm::clamp(camera_->pitch, -89.0f, 89.0f);
	}

	// 다음 프레임을 위해 현재 위치를 '이전 위치'로 저장한다.
	lastX = xpos;
	lastY = ypos;

	// --- 마우스 인터랙터 업데이트 로직 ---
	// 마우스 인터랙터에게는 가공되지 않은 순수한 좌표를 넘겨주는 역할만 한다.
	// 카메라 모드가 활성화되어 있든 아니든, 마우스의 현재 위치는 항상 알려줘야 한다.
	mouse_interactor_->mouse_pos_ = glm::vec2(static_cast<float>(xpos), static_cast<float>(ypos));
}

void Window::OnKey(int key, int scancode, int action, int mods) {
	// 키가 '눌리는 순간(PRESS)'에만 반응하도록 합니다.
	if (key == GLFW_KEY_F && action == GLFW_PRESS) {
		mouse_enabled_ = !mouse_enabled_;

		if (mouse_enabled_) {
			//glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			first_mouse_ = true; // 모드 전환 시 카메라 튀는 현상 방지
		}
		else {
			glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}
}

void Window::OnMouseClick(int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS)   mouse_interactor_->is_left_button_down_event = true;
		else if (action == GLFW_RELEASE) mouse_interactor_->is_left_button_up_event = true;
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		if (action == GLFW_PRESS)   mouse_interactor_->is_right_button_down_event = true;
		else if (action == GLFW_RELEASE) mouse_interactor_->is_right_button_up_event = true;
	}
}

Window::~Window()
{
	glfwDestroyWindow(glfw_window_);

	glfwTerminate();
}

void Window::run()
{
	double lastTime = glfwGetTime();

	while (!glfwWindowShouldClose(glfw_window_)) {
		double current = glfwGetTime();
		float dt = static_cast<float>(current - lastTime);
		lastTime = current;

		glfwPollEvents();
		ProcessKeyboard(dt);

		ctx_->Update(*camera_, *mouse_interactor_, dt);
		ctx_->Draw();
	}

	ctx_->WaitIdle();
}

void Window::ProcessKeyboard(float dt) {
	float v = camera_->move_speed * dt;

	// ASDW
	if (glfwGetKey(glfw_window_, GLFW_KEY_W) == GLFW_PRESS)
		camera_->position += camera_->Front() * v;
	if (glfwGetKey(glfw_window_, GLFW_KEY_S) == GLFW_PRESS)
		camera_->position -= camera_->Front() * v;
	if (glfwGetKey(glfw_window_, GLFW_KEY_A) == GLFW_PRESS)
		camera_->position -= camera_->Right() * v;
	if (glfwGetKey(glfw_window_, GLFW_KEY_D) == GLFW_PRESS)
		camera_->position += camera_->Right() * v;

	// QE
	if (glfwGetKey(glfw_window_, GLFW_KEY_Q) == GLFW_PRESS)
		camera_->position -= glm::vec3(0, 1, 0) * v;
	if (glfwGetKey(glfw_window_, GLFW_KEY_E) == GLFW_PRESS)
		camera_->position += glm::vec3(0, 1, 0) * v;

	// ← → 로 고개 좌/우 (yaw)
	const float yawSpeed = 120.0f; // deg/sec. 취향대로 60~180 사이로
	if (glfwGetKey(glfw_window_, GLFW_KEY_LEFT) == GLFW_PRESS) camera_->yaw -= yawSpeed * dt;
	if (glfwGetKey(glfw_window_, GLFW_KEY_RIGHT) == GLFW_PRESS) camera_->yaw += yawSpeed * dt;

	// ↑ ↓ 로 고개 위/아래 (pitch)도 원하면
	const float pitchSpeed = 90.0f; // deg/sec
	if (glfwGetKey(glfw_window_, GLFW_KEY_UP) == GLFW_PRESS) camera_->pitch = glm::clamp(camera_->pitch + pitchSpeed * dt, -89.0f, 89.0f);
	if (glfwGetKey(glfw_window_, GLFW_KEY_DOWN) == GLFW_PRESS) camera_->pitch = glm::clamp(camera_->pitch - pitchSpeed * dt, -89.0f, 89.0f);

	if (glfwGetKey(glfw_window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(glfw_window_, GLFW_TRUE);
	}
}