/**
 * @brief Wei Qi
 * @date 2024-7-20
 * @author Aurora
 */
#ifndef chess
#define chess
#include <array>

#define DEBUG

/**
 * @brief 定义一个复数类
 * @class Complex
 * @author Aurora
 * @date 2024-7-20
 */
class Complex {
public:
    Complex();
    Complex(int x, int y);

    /**
     * @brief 重载运算符
     * @param complex - 另一个向量
     * @return 向量和
     */
    Complex operator+(const Complex& complex) const;
    /**
     * @brief 重载运算符
     * @param complex - 另一个向量
     * @return 向量差
     */
    Complex operator-(const Complex& complex) const;

    int realPart = 0;
    int imaginaryPart = 0;
private:
};


/**
 * @brief 定义一个围棋棋盘
 * @author Aurora
 * @date 2024-7-20
 * @class Chessboard
 */
class Chessboard {
public:
    Chessboard();
    explicit Chessboard(std::array<std::array<int, 19>, 19> board);

    /**
     * @brief 落子
     * @param color - 0: 无棋子 1:黑棋 2:白棋
     * @param position - 棋盘坐标
     * @return 是否成功
     */
    bool drop(int color, Complex position);

    /**
     * @brief 清除改坐标上的子
     * @param position - 坐标
     * @return 是否成功
     */
    bool clear(Complex position);

    /**
     * @brief 打印棋子状态
     * @return 棋盘
     */
    std::array<std::array<int, 19>, 19> draw();

    /**
     * @brief 检测是否有棋子为死棋
     */
    void update();

    /**
     * @brief 获取棋子的颜色
     * @param position - 棋子坐标
     * @return 棋子的颜色 0: 无棋子 1:黑棋 2:白棋 3: ERROR
     */
    int getColor(Complex position);
private:
    std::array<std::array<int, 19>, 19> goBoard{};
};

#endif
