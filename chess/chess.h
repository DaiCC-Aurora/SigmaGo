/**
 * @brief Wei Qi
 * @date 2024-7-20
 * @author Aurora
 */
#ifndef chess
#define chess
#include <array>


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
 * @brief 棋子类
 * @class Chess
 * @author Aurora
 */
class Chess {
public:
    /**
     * @brief 构造函数 新建一个棋子
     * @param color : 棋子颜色 0: none -1: black 1: white
     * @param position : 棋子在复平面上的坐标
     */
    Chess(int color, Complex position);
    Chess();
    ~Chess();

    Complex position;   // 棋子坐标
    int color;  // 0: none -1: black 1: white
};

class Chessboard {
public:
    /**
     * @brief 初始化一个新棋盘
     */
    Chessboard();

    /**
     * @brief 落子
     * @param color : 棋子颜色
     * @param position : 棋子坐标
     */
    void drop(int color, Complex position);
    /**
     * @brief 查看颜色
     * @param position : 坐标
     * @return 该坐标的颜色
     */
    int getColor(Complex position);
    /**
     * @brief 更新棋盘状态
     */
    void update();

    /**
     * @return 获胜方 1: white -1: black 0: 平局
     */
    int whoWin();

    std::array<std::array<Chess, 19>, 19> chessboard;
protected:
    int whoTurn;    // 当前谁应该落子
    int sumChess;   // 当前共下棋子个数

private:
    bool DFS(const Complex& position, int color, std::vector<std::vector<bool>>& visited, std::vector<Complex>& group);
    int DFSForTerritory(const Complex& pos, std::vector<std::vector<bool>>&visited);
    std::array<Complex, 4> directions = {
            Complex(0, 1), // down
            Complex(0, -1), //up
            Complex(-1, 0),     // left
            Complex(1, 0)    // right
    };  // 表示方向
};

#endif
