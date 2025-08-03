import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, Eq, factor, solve, latex
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ─────────── Persist storage for the recognized polynomial ────────────────
if "recognized_expr" not in st.session_state:
    st.session_state.recognized_expr = None

# ─────────── Page setup ────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("✍️ Draw → Recognize → Solve / Integrate / Differentiate")

# ─────────── 1) Drawing Canvas ─────────────────────────────────────────────
st.header("1️⃣ Draw your curve (freehand)")
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

# ─────────── 2) Recognize Button ───────────────────────────────────────────
if st.button("🔍 Recognize"):
    pts = []
    if canvas_result.json_data and canvas_result.json_data["objects"]:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "path":
                for cmd in obj["path"]:
                    if cmd[0] in ("M", "L"):
                        pts.append(cmd[1:3])

    pts = np.array(pts)
    if len(pts) < 10:
        st.error("Please draw more of the curve before recognizing.")
    else:
        # invert y (canvas origin at top-left)
        X = pts[:, 0].reshape(-1, 1)
        Y = 300 - pts[:, 1]

        deg = st.slider("Polynomial degree fit", 1, 5, 2)
        poly = PolynomialFeatures(degree=deg)
        XP = poly.fit_transform(X)
        model = LinearRegression().fit(XP, Y)

        coeffs = model.coef_
        intercept = model.intercept_
        x = symbols('x')
        expr = intercept + sum(coeffs[i] * x**i for i in range(len(coeffs)))

        st.session_state.recognized_expr = sp.simplify(expr)
        st.success("Function recognized!")

# ─────────── 3) Show & Operate on Recognized Expr ──────────────────────────
expr = st.session_state.recognized_expr
if expr is not None:
    st.subheader("✅ Recognized function:")
    st.latex(r"f(x) = " + latex(expr))

    # Solve roots
    if st.button("🧮 Solve (roots)"):
        x = symbols('x')
        st.write("**Step 1: Set f(x)=0**")
        st.latex(latex(Eq(expr, 0)))

        st.write("**Step 2: Factor**")
        fac = factor(expr)
        st.latex(latex(Eq(expr, fac)))

        st.write("**Step 3: Solve each factor = 0**")
        try:
            facs = fac.as_ordered_factors()
        except:
            facs = [fac]
        for fct in facs:
            st.latex(latex(Eq(fct, 0)))

        st.write("**Step 4: Roots**")
        roots = solve(Eq(expr, 0), x)
        for r in roots:
            st.latex(latex(sp.Eq(x, r)))

    # Integrate
    if st.button("∫ Integrate"):
        x = symbols('x')
        integral = sp.integrate(expr, x)
        st.write("**∫ f(x) dx =**")
        st.latex(latex(integral))

    # Differentiate
    if st.button("d/dx Differentiate"):
        x = symbols('x')
        derivative = sp.diff(expr, x)
        st.write("**f '(x) =**")
        st.latex(latex(derivative))

    # Optional: plot vs sketch
    if st.button("📈 Plot vs. Sketch"):
        xs = np.linspace(pts[:,0].min(), pts[:,0].max(), 200)
        ys = [float(expr.subs(symbols('x'), v)) for v in xs]
        fig, ax = plt.subplots()
        ax.plot(pts[:,0], 300-pts[:,1], '.', label="Sketch")
        ax.plot(xs, ys, '-', label="Fit")
        ax.legend()
        st.pyplot(fig)
