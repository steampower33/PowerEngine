#include "model.h"
#include <limits> // std::numeric_limits ���

#include "ray.h"

Ray::Ray(const glm::vec3& origin, const glm::vec3& direction)
    : origin(origin), direction(glm::normalize(direction)) // �����ڿ��� �ٷ� ����ȭ
{
}

// �� �Լ��� �ٷ� �װ� �ø� RaySphereIntersect�� GLM �����̾�.
bool Ray::Intersects(const Model& model, float& dist) const
{
    // ������ ���������� ���� �߽��� ���ϴ� ����
    glm::vec3 oc = origin - model.position_;

    // 2�� �������� ��� ���
    // a = dot(D, D) : ���� ���� ������ ����. ����ȭ�Ǿ� �����Ƿ� �׻� 1.0
    const float a = 1.0f;

    // b = 2 * dot(OC, D)
    const float b = 2.0f * glm::dot(oc, direction);

    // c = dot(OC, OC) - r^2
    const float c = glm::dot(oc, oc) - model.radius_ * model.radius_;

    // �Ǻ��� (b^2 - 4ac)
    const float discriminant = b * b - 4.0f * a * c;

    // �Ǻ����� 0���� ������ �Ǳ��� �����Ƿ�, �浹���� ����
    if (discriminant < 0.0f) {
        return false;
    }

    // �� ���� �������� �浹 �Ÿ�(t)�� ���
    const float sqrt_discriminant = sqrt(discriminant);
    const float t0 = (-b - sqrt_discriminant) / (2.0f * a);
    const float t1 = (-b + sqrt_discriminant) / (2.0f * a);

    // ������ �� �������θ� ���ư��Ƿ�, t���� ������� ��.
    // �� ���� ��� t�� �� �� ���� ��(�� ����� ������)�� �����Ѵ�.

    float closest_t = std::numeric_limits<float>::max();
    bool hit = false;

    if (t0 > 0.0001f) { // ���� ���� ���� ���
        closest_t = t0;
        hit = true;
    }

    if (t1 > 0.0001f && t1 < closest_t) {
        closest_t = t1;
        hit = true;
    }

    if (hit) {
        dist = closest_t;
        return true;
    }

    return false;
}