import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import latex, Eq, factor, solve
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import traceback

# ─── 1) Build a local namespace containing *every* sympy name ──────────────
local_dict = {name: getattr(sp, name) for name in dir(sp) if not name.startswith("_")}
local_dict.update({'pi': sp.pi, 'E': sp.E})

# ─── 2) Streamlit page setup ──────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("✍️ Draw & Type → Recognize → Solve Math Problems")

# ─── 3) Choose input mode ─────────────────────────────────────────────────
mode = st.sidebar.selectbox("Input Mode", [
    "Draw Function (Polynomial)",
    "Type Math Expression"
])

# ─── 4) DRAW MODE ──────────────────────────────────────────────────────────
if mode == "Draw Function (Polynomial)":
    st.header("1️⃣ Draw a function curve (freehand)")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=3,
        stroke_color="#000000",
        background_color="#fff",
        height=300,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )

    recognized_expr = None
    if canvas_result.json_data and canvas_result.json_data["objects"]:
        # collect all (x,y) points from every path command
        pts = []
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "path":
                for cmd in obj["path"]:
                    if cmd[0] in ("M", "L"):
                        pts.append(cmd[1:3])
        pts = np.array(pts)
        if len(pts) > 10:
            x = pts[:, 0].reshape(-1, 1)
            y = 300 - pts[:, 1]   # invert y-axis for plotting

            deg = st.slider("Polynomial Degree Fit", 1, 5, 2)
            poly = PolynomialFeatures(degree=deg)
            Xp = poly.fit_transform(x)
            model = LinearRegression().fit(Xp, y)
            coeffs = model.coef_
            intercept = model.intercept_
            x_sym = sp.symbols('x')
            recognized_expr = intercept + sum(coeffs[i] * x_sym**i 
                                              for i in range(len(coeffs)))

            st.markdown("**Recognized Function:**")
            st.latex(r"f(x) = " + latex(recognized_expr))

    if recognized_expr is not None:
        st.header("2️⃣ Solve & Visualize")
        if st.button("Find Roots (f(x)=0)"):
            x = sp.symbols('x')
            st.subheader("Solution Steps")
            # Step 1
            st.write("1. Set f(x) = 0")
            st.latex(latex(Eq(recognized_expr, 0)))
            # Step 2
            fac = factor(recognized_expr)
            st.write("2. Factor the polynomial")
            st.latex(latex(Eq(recognized_expr, fac)))
            # Step 3
            try:
                factors = fac.as_ordered_factors()
            except:
                factors = [fac]
            st.write("3. Solve each factor = 0")
            for fct in factors:
                st.latex(latex(Eq(fct, 0)))
            # Step 4
            sols = solve(Eq(recognized_expr,0), x)
            st.write("4. Roots:")
            for sol in sols:
                st.latex(latex(sp.Eq(x, sol)))

        if st.button("Plot Recognized vs Sketch"):
            xs = np.linspace(pts[:,0].min(), pts[:,0].max(), 200)
            ys = [float(recognized_expr.subs(sp.symbols('x'), v)) for v in xs]
            fig, ax = plt.subplots()
            ax.plot(pts[:,0], 300-pts[:,1], '.', label="Sketch")
            ax.plot(xs, ys, '-', label="Fit")
            ax.legend()
            st.pyplot(fig)

# ─── 5) TEXT MODE ───────────────────────────────────────────────────────────
else:
    st.header("1️⃣ Type any math expression or command")
    st.markdown("""
Examples:
- `solve(x**2 - 2, x)`
- `integrate(sin(x), x)`
- `diff(x**3, x)`
- `limit(sin(x)/x, x, 0)`
- `simplify((x+1)**3)`
- `Matrix([[1,2],[3,4]])`
""")
    expr_input = st.text_area("Enter expression:", value="solve(x**2 - 2, x)")

    trans = standard_transformations + (implicit_multiplication_application,)

    def safe_parse(expr):
        try:
            return parse_expr(expr, local_dict=local_dict, transformations=trans)
        except Exception as e:
            return f"Parse Error: {e}"

    if st.button("Recognize & Solve"):
        parsed = safe_parse(expr_input)
        if isinstance(parsed, str):
            st.error(parsed)
        else:
            st.subheader("Parsed Object")
            st.write(parsed)
            st.latex(latex(parsed))

            st.subheader("Result")
            try:
                # If it’s a solve/integrate call, eval directly; else simplify
                if expr_input.strip().startswith(("solve", "integrate", "diff", "limit")):
                    result = eval(expr_input, {"__builtins__":None}, local_dict)
                else:
                    result = sp.simplify(parsed)
                st.write(result)
                st.latex(latex(result))
            except Exception:
                st.error("Evaluation Error:\n" + traceback.format_exc(limit=1))
