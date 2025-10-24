#include "camera.h"

glm::vec3 Camera::Front() const {
    float cy = cos(glm::radians(yaw));
    float sy = sin(glm::radians(yaw));
    float cp = cos(glm::radians(pitch));
    float sp = sin(glm::radians(pitch));

    //std::cout << position.x << " " << position.y << " " << position.z << std::endl;
    //std::cout << yaw << " " << pitch << " " << std::endl;
    return glm::normalize(glm::vec3(cy * cp, sp, sy * cp));
}
glm::vec3 Camera::Right() const {
    return glm::normalize(glm::cross(Front(), glm::vec3(0, 1, 0)));
}
glm::vec3 Camera::Up() const {
    return glm::normalize(glm::cross(Right(), Front()));
}

glm::mat4 Camera::View() const {
    return glm::lookAt(position, position + Front(), glm::vec3(0, 1, 0));
}
glm::mat4 Camera::Proj(float width, float height) const {
    glm::mat4 p = glm::perspective(glm::radians(fov), width / height, 0.1f, 1000.0f);
    // GLM은 OpenGL 기준으로 Y를 뒤집지 않지만, Vulkan 스크린 좌표계에 맞추려면 보통 아래 한 줄 추가
    p[1][1] *= -1; // Vulkan NDC 보정 (글꼴/좌표계 설정에 따라 제거 가능)
    return p;
}