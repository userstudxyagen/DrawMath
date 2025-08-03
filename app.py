import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, Eq, factor, solve, latex
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("✍️ Draw → Recognize → Solve")

# ── 1) Canvas for freehand drawing ─────────────────────────────────────────
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

# Prepare variables
recognized_expr = None
pts = []

# ── 2) Recognition Phase ─────────────────────────────────────────────────
if st.button("🔍 Recognize"):
    # extract all (x,y) points
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
        # invert y (canvas origin top-left)
        X = pts[:,0].reshape(-1,1)
        Y = (300 - pts[:,1])
        deg = st.slider("Choose polynomial degree for fit:", 1, 5, 2)
        poly = PolynomialFeatures(degree=deg)
        XP = poly.fit_transform(X)
        model = LinearRegression().fit(XP, Y)
        coeffs = model.coef_
        intercept = model.intercept_
        x = symbols('x')
        recognized_expr = intercept + sum(coeffs[i]*x**i for i in range(len(coeffs)))

        st.subheader("✅ Recognized Function")
        st.latex(r"f(x) = " + latex(recognized_expr))

# ── 3) Solve Phase ─────────────────────────────────────────────────────────
if recognized_expr is not None and st.button("🧮 Solve (find roots)"):
    x = symbols('x')
    st.header("Solution Steps")

    # Step 1: set f(x)=0
    st.write("**Step 1: Set the function equal to zero**")
    st.latex(latex(Eq(recognized_expr, 0)))

    # Step 2: factor
    st.write("**Step 2: Factor the polynomial**")
    factored = factor(recognized_expr)
    st.latex(latex(Eq(recognized_expr, factored)))

    # Step 3: set each factor to zero
    st.write("**Step 3: Set each factor = 0**")
    try:
        factors = factored.as_ordered_factors()
    except:
        factors = [factored]
    for fct in factors:
        st.latex(latex(Eq(fct, 0)))

    # Step 4: solve
    st.write("**Step 4: Solve for x**")
    roots = solve(Eq(recognized_expr, 0), x)
    for r in roots:
        st.latex(latex(sp.Eq(x, r)))

# ── 4) Optional: Plot comparison ───────────────────────────────────────────
if recognized_expr is not None and st.button("📈 Plot vs. Sketch"):
    xs = np.linspace(pts[:,0].min(), pts[:,0].max(), 200)
    ys = [float(recognized_expr.subs(symbols('x'), v)) for v in xs]
    fig, ax = plt.subplots()
    ax.plot(pts[:,0], 300-pts[:,1], '.', label="Your Sketch")
    ax.plot(xs, ys, '-', label="Fitted f(x)")
    ax.legend()
    st.pyplot(fig)
