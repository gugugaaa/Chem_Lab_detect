import cv2
def draw_landmarks(image, landmark_point):
    # 连接线
    if landmark_point is not None and len(landmark_point) > 0:
        # 大拇指
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                 (255, 255, 255), 2)

        # 食指
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                 (255, 255, 255), 2)

        # 中指
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                 (255, 255, 255), 2)

        # 无名指
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                 (255, 255, 255), 2)

        # 小指
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                 (255, 255, 255), 2)

        # 手掌
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                 (255, 255, 255), 2)

        # 关键点
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # 手腕1
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # 手腕2
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # 大拇指：根部
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # 大拇指：第1关节
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # 大拇指：指尖
                cv2.circle(image, (landmark[0], landmark[1]), 8,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # 食指：根部
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # 食指：第2关节
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # 食指：第1关节
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # 食指：指尖
                cv2.circle(image, (landmark[0], landmark[1]), 8,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # 中指：根部
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # 中指：第2关节
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # 中指：第1关节
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # 中指：指先
                cv2.circle(image, (landmark[0], landmark[1]), 8,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # 无名指：根部
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # 无名指：第2关节
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # 无名指：第1关节
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # 无名指：指尖
                cv2.circle(image, (landmark[0], landmark[1]), 8,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # 小指：根部
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # 小指：第2关节
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # 小指：第1关节
                cv2.circle(image, (landmark[0], landmark[1]), 5,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # 小指：指先
                cv2.circle(image, (landmark[0], landmark[1]), 8,
                           (255, 255, 255), -1)
                cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect and brect is not None:
        # 外接矩形
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                      (0, 0, 0), 1)

    return image


def draw_info_text(image, model_name, brect, handedness, hand_sign_text, fps):
    if brect is not None:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 30),
                      (0, 0, 0), -1)

    if handedness is not None:
        info_text = handedness.classification[0].label[0]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1,
                    cv2.LINE_AA)

    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, "Model:" + model_name, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "Model:" + model_name, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return image