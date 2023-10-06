import pygame
import math

vec2 = pygame.math.Vector2


def get_angle(vec):
    if vec.length() == 0:
        return 0
    return math.degrees(math.atan2(vec.y, vec.x))


def degrees_to_radians(angle):
    return angle / (180 / math.pi)


def radians_to_degrees(rads):
    return rads * 180 / math.pi


def box_line_collided(box, line):
    # check if any of the lines of the box collide with the line
    for i in range(len(box)):
        if lines_collided(
            box[i][0],
            box[i][1],
            box[(i + 1) % len(box)][0],
            box[(i + 1) % len(box)][1],
            line[0],
            line[1],
            line[2],
            line[3],
        ):
            return True
    return False


def lines_collided(x1, y1, x2, y2, x3, y3, x4, y4):
    uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (
        (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    )
    uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / (
        (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    )
    if 0 <= uA <= 1 and 0 <= uB <= 1:
        return True
    return False


def get_collision_point(x1, y1, x2, y2, x3, y3, x4, y4):
    global vec2
    uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (
        (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    )
    uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / (
        (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    )
    if 0 <= uA <= 1 and 0 <= uB <= 1:
        intersectionX = x1 + (uA * (x2 - x1))
        intersectionY = y1 + (uA * (y2 - y1))
        return vec2(intersectionX, intersectionY)
    return None


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def blit_rotate(surf, image, pos, originPos, angle):
    # offset from pivot to center
    image_rect = image.get_rect(topleft=(pos[0] - originPos[0], pos[1] - originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center

    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # rotated image center
    rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center=rotated_image_center)

    # rotate and blit the image
    surf.blit(rotated_image, rotated_image_rect)

    # draw rectangle around the image
    # pygame.draw.rect(
    #     surf, (255, 0, 0), (*rotated_image_rect.topleft, *rotated_image.get_size()), 2
    # )
