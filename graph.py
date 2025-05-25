import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import Eq, symbols
from typing import List

x, y = sp.symbols('x y')

def classify_expression(expr_str: str):
    try:
        expr_str = expr_str.replace("^", "**").strip()


        if '=' in expr_str and not any(op in expr_str for op in ['<', '>']):
            left, right = expr_str.split('=')
            left_expr = parse_expr(left.strip())
            right_expr = parse_expr(right.strip())
            expr = sp.Eq(left_expr, right_expr)
            return 'equation', expr


        if any(var in expr_str for var in ['y', 'Y']):

            if '<=' in expr_str:
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
        else:

            expr = parse_expr(expr_str)
            return 'function_x_only', expr

    except Exception:
        return 'invalid', None

def solve_for_y(expr):
    try:
        if isinstance(expr, Eq):
            expr = expr.lhs - expr.rhs
        solved = sp.solve(expr, y)
        return solved if solved else None
    except:
        return None

def solve_for_x(expr):
    try:
        if isinstance(expr, Eq):
            expr = expr.lhs - expr.rhs
        solved = sp.solve(expr, x)
        return solved if solved else None
    except:
        return None

def plot_function_y(expr, ax, color='green', label=None):
    try:
        y_func = sp.lambdify(x, expr, 'numpy')
        x_vals = np.linspace(-10, 10, 400)
        y_vals = y_func(x_vals)
        ax.plot(x_vals, y_vals, color=color, label=label or f'y = {expr}')
    except Exception as e:
        st.error(f"Function `{label}` error: {e}")

def plot_equation(expr, ax, color='blue', label=None):
    y_solutions = solve_for_y(expr)
    if y_solutions:
        for sol in y_solutions:
            plot_function_y(sol, ax, color=color, label=label)
    else:
        x_solutions = solve_for_x(expr)
        if x_solutions:
            for val in x_solutions:
                ax.axvline(float(val), color=color, linestyle='--', label=label or str(expr))
        else:
            st.warning(f"Could not solve equation `{expr}`")

def plot_inequality(expr, ax, color='lightblue', label=None):
    try:
        y_solutions = solve_for_y(expr)
        if not y_solutions:
            st.warning(f"Could not solve inequality `{expr}`")
            return
        y_expr = y_solutions[0]
        y_func = sp.lambdify(x, y_expr, 'numpy')
        x_vals = np.linspace(-10, 10, 400)
        y_vals = y_func(x_vals)
        linestyle = '--' if isinstance(expr, (sp.StrictLessThan, sp.StrictGreaterThan)) else '-'
        ax.plot(x_vals, y_vals, color='black', linestyle=linestyle, label=label or str(expr))
        X, Y = np.meshgrid(x_vals, np.linspace(-10, 10, 400))
        f = sp.lambdify((x, y), expr, 'numpy')
        region = f(X, Y)
        ax.contourf(X, Y, region, levels=[0.5, 1], colors=[color], alpha=0.5, zorder=-1)
    except Exception as e:
        st.error(f"Could not plot inequality `{expr}`: {e}")

def plot_function_x_only(expr, ax, color='purple', label=None):
    # For expressions only in x (like polynomials), plot y = expr(x)
    try:
        y_func = sp.lambdify(x, expr, 'numpy')
        x_vals = np.linspace(-10, 10, 400)
        y_vals = y_func(x_vals)
        ax.plot(x_vals, y_vals, color=color, label=label or f'y = {expr}')
        # If polynomial == 0, mark roots on x-axis
        roots = sp.solve(expr, x)
        for r in roots:
            if r.is_real:
                ax.plot(float(r), 0, 'ro')
                ax.annotate(f'{float(r):.2f}', (float(r), 0), textcoords="offset points", xytext=(5,-10))
    except Exception as e:
        st.error(f"Error plotting polynomial `{label}`: {e}")

def find_intersections(exprs):
    intersections = []
    equations = [classify_expression(e)[1] for e in exprs if classify_expression(e)[0] == 'equation']
    if len(equations) >= 2:
        try:
            sol = sp.solve(equations[:2], (x, y), dict=True)
            intersections = sol if sol else []
        except:
            pass
    return intersections

def create_plot(exprs: List[str], xlim=(-10, 10), ylim=(-10, 10)):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

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
        elif expr_type == 'function_x_only':
            plot_function_x_only(parsed_expr, ax, color, label=expr_str)
        else:
            st.warning(f"Expression not handled: {expr_str}")

    intersections = find_intersections(exprs)
    for point in intersections:
        x_val = float(point[x])
        y_val = float(point[y])
        ax.plot(x_val, y_val, 'ko')
        ax.annotate(f"({x_val:.2f}, {y_val:.2f})", (x_val, y_val), textcoords="offset points", xytext=(5,5))

    ax.legend()
    return fig

def main():
    st.set_page_config(page_title="Graphic Calculator", layout="wide")
    st.title("Graphing Calculator")
    st.markdown("Supports: `functions`, `equations`, `inequalities`. Examples: `y = 2*x + 1`, `x + y = 4`, `y < -x + 3`, `x^2 + y^2 = 25`, `x = 3`, `6*x**2 + 5*x - 4`, `6*x**2 + 5*x - 4 = 0`")

    with st.expander("Graph Settings"):
        xlim = st.slider("X-axis range", -50, 50, (-10, 10))
        ylim = st.slider("Y-axis range", -50, 50, (-10, 10))

    st.subheader("Enter up to 5 expressions:")
    col1, col2 = st.columns(2)
    expressions = []
    for i in range(5):
        with col1 if i % 2 == 0 else col2:
            expr = st.text_input(f"Expression {i+1}", value="", key=f"expr_{i}")
            if expr.strip():
                expressions.append(expr)

    if expressions:
        fig = create_plot(expressions, xlim=xlim, ylim=ylim)
        st.pyplot(fig)
        st.download_button("Download Graph as PNG", data=fig_to_bytes(fig), file_name="graph.png", mime="image/png")
    else:
        st.info("Enter an expression above to see the graph.")

def fig_to_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf.read()

if __name__ == '__main__':
    main()
