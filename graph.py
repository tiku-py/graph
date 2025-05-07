

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import Eq, symbols, solveset, S
from typing import List


x, y = sp.symbols('x y')


def classify_expression(expr_str: str):
    try:
        expr_str = expr_str.replace("^", "**").strip()


        if '=' in expr_str and not any(op in expr_str for op in ['<', '>']):
            left, right = expr_str.split('=')
            expr = sp.Eq(parse_expr(left.strip()), parse_expr(right.strip()))
            return 'equation', expr
        elif '<=' in expr_str:
            left, right = expr_str.split('<=')
            expr = parse_expr(left.strip()) <= parse_expr(right.strip())
            return 'inequality', expr
        elif '>=' in expr_str:
            left, right = expr_str.split('>=')
            expr = parse_expr(left.strip()) >= parse_expr(right.strip())
            return 'inequality', expr
        elif '<' in expr_str:
            left, right = expr_str.split('<')
            expr = parse_expr(left.strip()) < parse_expr(right.strip())
            return 'inequality', expr
        elif '>' in expr_str:
            left, right = expr_str.split('>')
            expr = parse_expr(left.strip()) > parse_expr(right.strip())
            return 'inequality', expr
        else:

            expr = parse_expr(expr_str)
            return 'function_y', expr
    except Exception as e:
        return 'invalid', None


def solve_for_y(expr):

    try:

        if expr.lhs == y:
            return expr.rhs
        elif expr.rhs == y:
            return expr.lhs
        else:

            solved = sp.solve(expr, y)
            return solved[0] if solved else None
    except:
        return None


def plot_function_y(expr, ax, color='green', label=None):
    try:
        y_func = sp.lambdify(x, expr, 'numpy')
        x_vals = np.linspace(-10, 10, 400)
        y_vals = y_func(x_vals)
        ax.plot(x_vals, y_vals, color=color, label=label or f'y = {expr}')
    except Exception as e:
        st.error(f"Function error: {e}")


def plot_equation(expr, ax, color='blue', label=None):
    y_expr = solve_for_y(expr)
    if y_expr:
        plot_function_y(y_expr, ax, color=color, label=label or str(expr))
    else:
        st.warning(f"Could not solve equation: {expr}")


def plot_inequality(expr, ax, color='lightblue', label=None):
    y_expr = solve_for_y(expr)
    if not y_expr:
        st.warning(f"Could not solve inequality: {expr}")
        return
    try:
        y_func = sp.lambdify(x, y_expr, 'numpy')
        x_vals = np.linspace(-10, 10, 400)
        y_vals = y_func(x_vals)

        # Plot boundary
        linestyle = '--' if isinstance(expr, (sp.StrictLessThan, sp.StrictGreaterThan)) else '-'
        ax.plot(x_vals, y_vals, color='black', linestyle=linestyle, label=label or str(expr))

        # Shading
        X, Y = np.meshgrid(x_vals, np.linspace(-10, 10, 400))
        f = sp.lambdify((x, y), expr, 'numpy')
        region = f(X, Y)
        ax.contourf(X, Y, region, levels=[0.5, 1], colors=[color], alpha=0.5)
    except Exception as e:
        st.error(f"Inequality error: {e}")


def create_plot(exprs: List[str]):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    colors = ['green', 'blue', 'red', 'purple', 'orange']
    for i, expr_str in enumerate(exprs):
        expr_type, parsed_expr = classify_expression(expr_str)
        if expr_type == 'invalid':
            st.error(f"Invalid expression: {expr_str}")
            continue

        color = colors[i % len(colors)]

        if expr_type == 'function_y':
            plot_function_y(parsed_expr, ax, color, label=expr_str)
        elif expr_type == 'equation':
            plot_equation(parsed_expr, ax, color, label=expr_str)
        elif expr_type == 'inequality':
            plot_inequality(parsed_expr, ax, label=expr_str)
        else:
            st.warning(f"Expression not handled: {expr_str}")

    ax.legend()
    return fig

# Streamlit UI
def main():
    st.set_page_config(page_title="Desmos Clone", layout="centered")
    st.title("📊 Graphing Calculator")
    st.markdown("Supports: `functions`, `equations`, `inequalities`. Examples: `y = 2*x + 1`, `x + y = 4`, `y < -x + 3`")

    st.subheader("Enter up to 5 expressions:")
    expressions = []
    for i in range(5):
        expr = st.text_input(f"Expression {i+1}", value="", key=f"expr_{i}")
        if expr.strip():
            expressions.append(expr)

    if expressions:
        fig = create_plot(expressions)
        st.pyplot(fig)
    else:
        st.info("Enter an expression above to see the graph.")

    st.markdown("---")
    

if __name__ == '__main__':
    main()
