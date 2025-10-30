#pragma once

class Model;

struct BoundingSphere
{
    glm::vec3 center{ 0.0f };
    float radius = 0.0f;
};

class Ray
{
public:
    // ������: ���� ���ʹ� ���� �� �ڵ����� ����ȭ(normalize)�ȴ�.
    Ray(const glm::vec3& origin, const glm::vec3& direction);

    // ������ ���� �浹�� �˻��ϴ� ��� �Լ�
    // dist: [��¿�] �浹 ��, ���� ������������ �Ÿ��� ���� ����
    // ��ȯ��: �浹 ����
    bool Intersects(const Model& model, float& dist) const;

public:
    glm::vec3 origin;
    glm::vec3 direction; // �׻� ����ȭ�� ����
};