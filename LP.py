import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from pathlib import Path

draggable_iso_profit = components.declare_component(
    "draggable_iso_profit_lpc",
    path=str(Path(__file__).parent / "draggable_iso_profit")
)

# -------------------------------------------------------------------------------------------------
# 通用輔助函式：整數顯示整數，浮點數顯示至小數點後兩位
# -------------------------------------------------------------------------------------------------
def fmt_num(val):
    if abs(val - round(val)) < 1e-5:
        return f"{int(round(val))}"
    return f"{val:.2f}"

# -------------------------------------------------------------------------------------------------
# 頁面基本配置與現代風格排版
# -------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="線性規劃與作業研究決策支援系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }
    h1, h2, h3, h4 {
        font-weight: 600 !important;
        color: #0f172a !important;
        letter-spacing: -0.02em;
    }
    .stMetric {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =================================================================================================
# 核心求解器模組 (採用標準 SciPy HiGHS 引擎)
# =================================================================================================
def solve_lp_highs(opt_type, c, constraints, non_neg=True):
    """
    使用 SciPy HiGHS 求解通用線性規劃，支援影子價格與嚴謹狀態判定。
    """
    n = len(c)
    c_factor = -1.0 if opt_type == 'Max' else 1.0
    c_arr = np.array(c, dtype=float) * c_factor

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    for con in constraints:
        a = list(map(float, con['a']))
        rhs = float(con['rhs'])
        if con['op'] == '<=':
            A_ub.append(a)
            b_ub.append(rhs)
        elif con['op'] == '>=':
            A_ub.append([-val for val in a])
            b_ub.append(-rhs)
        elif con['op'] == '=':
            A_eq.append(a)
            b_eq.append(rhs)

    bounds = [(0, None) if non_neg else (None, None) for _ in range(n)]

    res = linprog(
        c=c_arr,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method='highs'
    )

    if not res.success:
        return False, None, res.message

    best_x = res.x
    best_z = res.fun * c_factor

    shadow_prices = []
    ub_idx = 0
    eq_idx = 0

    dual_ub = res.ineqlin.marginals if (hasattr(res, 'ineqlin') and res.ineqlin is not None) else []
    dual_eq = res.eqlin.marginals if (hasattr(res, 'eqlin') and res.eqlin is not None) else []

    for con in constraints:
        if con['op'] == '<=':
            val = dual_ub[ub_idx] if len(dual_ub) > ub_idx else 0.0
            sp = abs(val) if opt_type == 'Max' else -abs(val)
            shadow_prices.append(sp)
            ub_idx += 1
        elif con['op'] == '>=':
            val = dual_ub[ub_idx] if len(dual_ub) > ub_idx else 0.0
            sp = -abs(val) if opt_type == 'Max' else abs(val)
            shadow_prices.append(sp)
            ub_idx += 1
        elif con['op'] == '=':
            val = dual_eq[eq_idx] if len(dual_eq) > eq_idx else 0.0
            sp = -val if opt_type == 'Max' else val
            shadow_prices.append(sp)
            eq_idx += 1

    return True, {
        'x': best_x,
        'z': best_z,
        'shadow_prices': shadow_prices,
        'status_msg': res.message
    }, ""

# =================================================================================================
# 側邊欄導航 (Navigation)
# =================================================================================================
app_mode = st.sidebar.radio(
    "系統功能導覽",
    [
        "線性規劃計算器",
        "運輸問題排程系統",
        "薪資最佳化決策支援系統",
        "可自訂限制式參數之薪資決策系統"
    ]
)

# -------------------------------------------------------------------------------------------------
# 頁面一：通用線性規劃計算器
# -------------------------------------------------------------------------------------------------
if app_mode == "線性規劃計算器":
    st.title("線性規劃計算器 (LP Calculator)")
    st.caption("支援多決策變數建模、即時數學模型顯示、敏感度與影子價格分析、2D 幾何與等利潤線視覺化")

    st.subheader("1. 模型基本設定與目標函數")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        opt_type = st.selectbox("最佳化類型", ["Max", "Min"])
    
    with c2:
        num_vars = st.number_input(
            "決策變數數量",
            min_value=1,
            max_value=6,
            value=2,
            step=1
        )
    
    st.markdown("**請輸入目標函數係數 (Z)：**")
    obj_cols = st.columns(num_vars)
    obj_c = []
    default_c = [20.0, 15.0, 10.0, 10.0, 10.0, 10.0]
    
    for i in range(num_vars):
        with obj_cols[i]:
            val = st.number_input(
                f"x{i+1} 係數",
                value=default_c[i] if i < len(default_c) else 1.0,
                key=f"obj_{i}"
            )
            obj_c.append(val)

    st.markdown("---")
    st.subheader("2. 限制式條件設定")
    non_neg = st.checkbox("啟用非負限制 (xᵢ ≥ 0)", value=True)
    
    num_constraints = st.number_input("限制式數量", min_value=1, max_value=8, value=2, step=1)

    constraints_input = []
    default_A = [[2.0, 1.0], [1.0, 3.0]]
    default_rhs = [20.0, 35.0]

    for k in range(num_constraints):
        st.markdown(f"**限制式 ({k+1})：**")
        c_cols = st.columns(num_vars + 2)
        row_a = []
        for j in range(num_vars):
            with c_cols[j]:
                init_val = default_A[k][j] if (k < len(default_A) and j < len(default_A[k])) else 1.0
                val = st.number_input(f"x{j+1}", value=float(init_val), key=f"c_{k}_{j}")
                row_a.append(val)
        with c_cols[num_vars]:
            op = st.selectbox("符號", ["<=", ">=", "="], index=0, key=f"op_{k}")
        with c_cols[num_vars + 1]:
            rhs_init = default_rhs[k] if k < len(default_rhs) else 10.0
            rhs_val = st.number_input("右端常數 (RHS)", value=float(rhs_init), key=f"rhs_{k}")

        constraints_input.append({'a': row_a, 'op': op, 'rhs': rhs_val})

    st.markdown("---")
    st.subheader("3. 數學模型即時預覽")
    obj_latex = " + ".join([f"{fmt_num(obj_c[i])}x_{{{i+1}}}" for i in range(num_vars)])
    latex_str = f"\\begin{{aligned}} \\text{{{opt_type}}}\\quad Z &= {obj_latex} \\\\\n\\text{{subject to}}\\quad "
    con_latex_list = []
    for k, con in enumerate(constraints_input):
        lhs_terms = " + ".join([f"{fmt_num(con['a'][j])}x_{{{j+1}}}" for j in range(num_vars)])
        sym = "\\le" if con['op'] == '<=' else ("\\ge" if con['op'] == '>=' else "=")
        con_latex_list.append(f"{lhs_terms} &{sym} {fmt_num(con['rhs'])}")
    if non_neg:
        vars_str = ", ".join([f"x_{{{j+1}}}" for j in range(num_vars)])
        con_latex_list.append(f"{vars_str} &\\ge 0")
    latex_str += " \\\\\n".join(con_latex_list) + "\n\\end{aligned}"
    st.latex(latex_str)
    dragged_z = st.session_state.get("iso_profit_drag_value")
    if num_vars == 2 and dragged_z is not None:
        st.latex(
            f"Z = {fmt_num(obj_c[0])}x_1 + {fmt_num(obj_c[1])}x_2 = {fmt_num(dragged_z)}"
        )

    solve_clicked = st.button("求解最佳解 (Solve)", type="primary")
    if solve_clicked:
        st.session_state["lp_solved"] = True
    if solve_clicked or st.session_state.get("lp_solved", False):
        success, sol_data, err_msg = solve_lp_highs(opt_type, obj_c, constraints_input, non_neg)

        if not success:
            st.error(f"求解失敗：此模型無可行解 (Infeasible) 或解為無界解 (Unbounded)。詳細訊息: {err_msg}")
        else:
            sol = sol_data['x']
            best_z = sol_data['z']
            s_prices = sol_data['shadow_prices']

            st.success("求解成功！已找到最佳決策方案。")
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.metric("最佳目標函數值 (Z*)", fmt_num(best_z))
            with res_c2:
                lines = [f"x{j+1} = {fmt_num(sol[j])}" for j in range(num_vars)]
                sol_multiline = "<br>".join(lines)
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px 18px; border-radius: 8px;">
                    <div style="font-size: 13px; color: #64748b; margin-bottom: 4px;">最佳決策變數方案</div>
                    <div style="font-size: 20px; font-weight: 600; color: #0f172a; line-height: 1.5;">
                        {sol_multiline}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("4. 最佳決策方案之管理意涵解讀")
            meaning_parts = [f"生產/配置決策變數 x{j+1} 為 **{fmt_num(sol[j])}** 單位" for j in range(num_vars)]
            meaning_text = (
                f"根據線性規劃求解結果，建議之最佳執行方案為：{'； '.join(meaning_parts)}。"
                f"在此資源分配條件下，預計可達成最佳目標總值 (Z*) 為 **{fmt_num(best_z)}**。"
            )
            st.info(meaning_text)

            st.subheader("5. 限制式與敏感度分析")
            table_data = []
            interpretations = []

            for k, con in enumerate(constraints_input):
                lhs_val = sum(con['a'][j] * sol[j] for j in range(num_vars))
                diff = abs(lhs_val - con['rhs'])
                is_binding = diff < 1e-4
                slack = 0.0 if is_binding else abs(con['rhs'] - lhs_val)
                sp = s_prices[k]

                if is_binding:
                    status_str = "Binding (緊束瓶頸)"
                    if abs(sp) > 1e-4:
                        sign = "+" if sp >= 0 else ""
                        interp = f"**限制式 ({k+1})** 為瓶頸限制。在其他條件不變下，右端常數 (RHS) 每增加 1 單位， Z 預計改變 {sign}{fmt_num(sp)} 單位。"
                    else:
                        interp = f"**限制式 ({k+1})** 為緊束限制 (等號成立)，但目前邊際影子價格為 0 (可能存在退化或多重解)。"
                else:
                    status_str = "Nonbinding (有餘量)"
                    interp = f"**限制式 ({k+1})** 尚有未利用之餘裕量 {fmt_num(slack)} 單位，增加此項資源短期內不會改變最佳解。"

                interpretations.append(interp)
                table_data.append({
                    "限制式": f"限制式 ({k+1})",
                    "左側值 (LHS)": fmt_num(lhs_val),
                    "運算符號": con['op'],
                    "右側常數 (RHS)": fmt_num(con['rhs']),
                    "鬆弛/剩餘量 (Slack/Surplus)": fmt_num(slack),
                    "狀態判定": status_str,
                    "影子價格 (Shadow Price)": fmt_num(sp)
                })

            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            with st.expander("影子價格 (Shadow Price) 管理決策意涵說明", expanded=True):
                for interp in interpretations:
                    st.write(f"• {interp}")
                st.warning(
                    "⚠️ **作業研究理論提醒**：上述影子價格 (Shadow Price) 為局部邊際變化率，"
                    "僅在當前最佳基底 (Optimal Basis) 之右端項允許變動範圍 (Allowable Range) 內有效。"
                    "若 RHS 調整幅度超出基底不變之範圍，影子價格與瓶頸結構將會改變。"
                )

            if num_vars == 2:
                st.markdown("---")
                st.subheader("6. 二維幾何圖形與等利潤線視覺化 (2D Graphic Analysis)")
                fig = go.Figure()

                max_x = max(25.0, sol[0] * 1.8)
                max_y = max(25.0, sol[1] * 1.8)

                x_vals = np.linspace(0, max_x, 300)
                for k, con in enumerate(constraints_input):
                    a1, a2 = con['a'][0], con['a'][1]
                    rhs = con['rhs']
                    if abs(a2) > 1e-5:
                        y_vals = (rhs - a1 * x_vals) / a2
                        mask = (y_vals >= 0) & (y_vals <= max_y)
                        fig.add_trace(go.Scatter(
                            x=x_vals[mask], y=y_vals[mask],
                            mode='lines',
                            name=f"限制式 ({k+1}): {fmt_num(a1)}x₁ + {fmt_num(a2)}x₂ {con['op']} {fmt_num(rhs)}"
                        ))
                    elif abs(a1) > 1e-5:
                        x_const = rhs / a1
                        fig.add_trace(go.Scatter(
                            x=[x_const, x_const], y=[0, max_y],
                            mode='lines',
                            name=f"限制式 ({k+1}): x₁ = {fmt_num(x_const)}"
                        ))

                all_ineqs = []
                for con in constraints_input:
                    all_ineqs.append({'a': list(con['a']), 'op': con['op'], 'rhs': float(con['rhs'])})
                if non_neg:
                    all_ineqs.append({'a': [1.0, 0.0], 'op': '>=', 'rhs': 0.0})
                    all_ineqs.append({'a': [0.0, 1.0], 'op': '>=', 'rhs': 0.0})
                all_ineqs.append({'a': [1.0, 0.0], 'op': '<=', 'rhs': max_x})
                all_ineqs.append({'a': [0.0, 1.0], 'op': '<=', 'rhs': max_y})

                cand_pts = []
                for i_idx in range(len(all_ineqs)):
                    for j_idx in range(i_idx + 1, len(all_ineqs)):
                        A_mat = [all_ineqs[i_idx]['a'], all_ineqs[j_idx]['a']]
                        b_vec = [all_ineqs[i_idx]['rhs'], all_ineqs[j_idx]['rhs']]
                        try:
                            pt = np.linalg.solve(A_mat, b_vec)
                            is_feas = True
                            for ineq in all_ineqs:
                                v = ineq['a'][0] * pt[0] + ineq['a'][1] * pt[1]
                                if ineq['op'] == '<=' and v > ineq['rhs'] + 1e-4:
                                    is_feas = False; break
                                elif ineq['op'] == '>=' and v < ineq['rhs'] - 1e-4:
                                    is_feas = False; break
                                elif ineq['op'] == '=' and abs(v - ineq['rhs']) > 1e-4:
                                    is_feas = False; break
                            if is_feas:
                                cand_pts.append(pt)
                        except np.linalg.LinAlgError:
                            continue

                if len(cand_pts) > 0:
                    cand_pts = np.unique(np.array(cand_pts).round(decimals=3), axis=0)

                    drawn = False
                    if len(cand_pts) >= 3:
                        try:
                            hull = ConvexHull(cand_pts)
                            hull_pts = cand_pts[hull.vertices]
                            fig.add_trace(go.Scatter(
                                x=list(hull_pts[:, 0]) + [hull_pts[0, 0]],
                                y=list(hull_pts[:, 1]) + [hull_pts[0, 1]],
                                fill="toself",
                                fillcolor="rgba(34, 197, 94, 0.25)",
                                line=dict(color="rgba(34, 197, 94, 0.8)", width=1.5),
                                name="可行解區域 (Feasible Region)"
                            ))
                            drawn = True
                        except Exception:
                            drawn = False

                    if not drawn:
                        if len(cand_pts) >= 2:
                            cand_pts = cand_pts[np.lexsort((cand_pts[:, 1], cand_pts[:, 0]))]
                            fig.add_trace(go.Scatter(
                                x=cand_pts[:, 0],
                                y=cand_pts[:, 1],
                                mode='lines+markers',
                                line=dict(color="rgba(34, 197, 94, 0.9)", width=4),
                                marker=dict(size=6, color="rgba(34, 197, 94, 0.9)"),
                                name="可行解線段 (Feasible Line Segment)"
                            ))
                        elif len(cand_pts) == 1:
                            fig.add_trace(go.Scatter(
                                x=[cand_pts[0][0]],
                                y=[cand_pts[0][1]],
                                mode='markers',
                                marker=dict(size=10, color="green", symbol="diamond"),
                                name="唯一可行解點 (Feasible Point)"
                            ))

                c_x1, c_x2 = obj_c[0], obj_c[1]
                z_corners = [
                    c_x1 * x + c_x2 * y
                    for x in (0.0, max_x)
                    for y in (0.0, max_y)
                ]
                z_min, z_max = min(z_corners), max(z_corners)
                if abs(z_max - z_min) < 1e-9:
                    z_min, z_max = best_z - 1.0, best_z + 1.0
                z_value = float(np.clip(
                    st.session_state.get("iso_profit_drag_value", best_z), z_min, z_max
                ))
                if abs(c_x2) > 1e-5:
                    y_obj = (z_value - c_x1 * x_vals) / c_x2
                    mask_obj = (y_obj >= 0) & (y_obj <= max_y)
                    fig.add_trace(go.Scatter(
                        x=x_vals[mask_obj], y=y_obj[mask_obj],
                        mode='lines',
                        line=dict(color='purple', width=2.5, dash='dash'),
                        name=f"等利潤線 (Z = {fmt_num(z_value)})"
                    ))
                elif abs(c_x1) > 1e-5:
                    x_obj = z_value / c_x1
                    if 0 <= x_obj <= max_x:
                        fig.add_trace(go.Scatter(
                            x=[x_obj, x_obj], y=[0, max_y], mode='lines',
                            line=dict(color='purple', width=2.5, dash='dash'),
                            name=f"等利潤線 (Z = {fmt_num(z_value)})"
                        ))

                fig.add_trace(go.Scatter(
                    x=[sol[0]], y=[sol[1]],
                    mode='markers+text',
                    name='最佳解點 (Optimal Solution)',
                    marker=dict(color='red', size=14, symbol='star'),
                    text=[f"最佳解 ({fmt_num(sol[0])}, {fmt_num(sol[1])})"],
                    textposition="top right"
                ))

                fig.update_layout(
                    title="線性規劃可行區域、等利潤線與最佳解幾何點位",
                    xaxis=dict(title="決策變數 x₁", range=[0, max_x]),
                    yaxis=dict(title="決策變數 x₂", range=[0, max_y]),
                    height=480
                )
                if abs(c_x1) + abs(c_x2) < 1e-10:
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("目前目標函數係數皆為 0，沒有可拖曳的等利潤線。")
                else:
                    draggable_iso_profit(
                        figure_json=fig.to_json(),
                        objective=[float(c_x1), float(c_x2)],
                        z_value=z_value,
                        min_z=float(z_min),
                        max_z=float(z_max),
                        key="iso_profit_drag_value",
                        default=float(best_z),
                    )

# -------------------------------------------------------------------------------------------------
# 頁面二：進階範本一 - 運輸問題排程系統
# -------------------------------------------------------------------------------------------------
elif app_mode == "運輸問題排程系統":
    st.title("運輸問題排程決策系統 (Transportation Problem)")
    st.caption("經典作業研究應用：多產地與多銷地間的供需平衡與物流最低成本配置模型")

    st.subheader("Step 1 & 2: 運輸模型規模設定")
    c1, c2 = st.columns(2)
    with c1:
        num_sources = st.number_input("供應地數量 (Sources / 供給地)", min_value=1, max_value=6, value=3, step=1)
    with c2:
        num_dests = st.number_input("需求地數量 (Destinations / 需求地)", min_value=1, max_value=6, value=4, step=1)

    st.subheader("Step 3: 輸入供應量、需求量與單位運費")
    supp_cols = st.columns(num_sources)
    supplies = []
    st.markdown("**各供給地供應量 (Supply)：**")
    for i in range(num_sources):
        with supp_cols[i]:
            val = st.number_input(f"供應地 S{i+1}", min_value=0.0, value=200.0, step=10.0, key=f"supp_{i}")
            supplies.append(val)

    dem_cols = st.columns(num_dests)
    demands = []
    st.markdown("**各需求地目標需求量 (Demand)：**")
    for j in range(num_dests):
        with dem_cols[j]:
            val = st.number_input(f"需求地 D{j+1}", min_value=0.0, value=150.0, step=10.0, key=f"dem_{j}")
            demands.append(val)

    st.markdown("**單位運費矩陣 (Unit Cost Matrix)：**")
    default_grid = []
    for i in range(num_sources):
        row = []
        for j in range(num_dests):
            row.append(float((i + 1) * 3 + (j + 1) * 2))
        default_grid.append(row)

    df_cost_input = pd.DataFrame(
        default_grid,
        index=[f"供應地 S{i+1}" for i in range(num_sources)],
        columns=[f"需求地 D{j+1}" for j in range(num_dests)]
    )
    edited_costs = st.data_editor(df_cost_input, use_container_width=True)

    # Step 4 & 5：自動建立與顯示完整線性規劃數學模型 (改為單一流水號 x1, x2, ...)
    st.subheader("Step 4 & 5: 自動建立與顯示線性規劃數學模型")
    
    col_opt, _ = st.columns([2, 3])
    with col_opt:
        expand_all_latex = st.checkbox("完整展開目標函數所有變數項", value=(num_sources * num_dests <= 8))

    with st.expander("點擊查看系統自動生成之完整 LP 數學模型", expanded=True):
        cost_mat = edited_costs.values
        total_terms = num_sources * num_dests

        # 目標函數：改為 x1, x2, x3, ...
        obj_terms = []
        k = 1
        for i in range(num_sources):
            for j in range(num_dests):
                obj_terms.append(f"{fmt_num(cost_mat[i][j])}x_{{{k}}}")
                k += 1

        if expand_all_latex or total_terms <= 8:
            chunk_size = 4
            chunks = [" + ".join(obj_terms[idx:idx + chunk_size]) for idx in range(0, len(obj_terms), chunk_size)]
            obj_str = " \\\\\n&\\quad + ".join(chunks)
        else:
            obj_str = " + ".join(obj_terms[:6]) + " + \\cdots + " + obj_terms[-1]

        lp_display = f"\\begin{{aligned}} \\min\\quad Z &= {obj_str} \\\\\n\\text{{subject to}}\\quad "

        con_lines = []
        # 1. 供給限制式 (每一行依序為 x1~x4, x5~x8, ...)
        for i in range(num_sources):
            row_vars = [f"x_{{{i * num_dests + j + 1}}}" for j in range(num_dests)]
            con_lines.append(f"{' + '.join(row_vars)} &\\le {fmt_num(supplies[i])}")

        # 2. 需求限制式 (縱向抓取變數)
        for j in range(num_dests):
            col_vars = [f"x_{{{i * num_dests + j + 1}}}" for i in range(num_sources)]
            con_lines.append(f"{' + '.join(col_vars)} &\\ge {fmt_num(demands[j])}")

        vars_all = ", ".join([f"x_{{{m+1}}}" for m in range(total_terms)])
        if total_terms <= 6:
            con_lines.append(f"{vars_all} &\\ge 0")
        else:
            con_lines.append("x_k &\\ge 0, \\quad \\forall k=1,\\dots," + str(total_terms))

        lp_display += " \\\\\n".join(con_lines) + "\n\\end{aligned}"
        st.latex(lp_display)

    if st.button("Step 6: 求解最佳調度方案 (Solve LP)", type="primary"):
        costs_matrix = edited_costs.values.tolist()
        num_vars_tp = num_sources * num_dests
        c_tp = []
        for i in range(num_sources):
            for j in range(num_dests):
                c_tp.append(costs_matrix[i][j])

        A_ub_tp = []
        b_ub_tp = []

        for i in range(num_sources):
            row = [0.0] * num_vars_tp
            for j in range(num_dests):
                row[i * num_dests + j] = 1.0
            A_ub_tp.append(row)
            b_ub_tp.append(supplies[i])

        for j in range(num_dests):
            row = [0.0] * num_vars_tp
            for i in range(num_sources):
                row[i * num_dests + j] = -1.0
            A_ub_tp.append(row)
            b_ub_tp.append(-demands[j])

        bounds_tp = [(0, None) for _ in range(num_vars_tp)]
        res_tp = linprog(c_tp, A_ub=A_ub_tp, b_ub=b_ub_tp, bounds=bounds_tp, method='highs')

        if not res_tp.success:
            st.error(f"運輸模型求解失敗：無法滿足供需條件（{res_tp.message}）。請檢查總供給是否小於總需求。")
        else:
            total_cost = res_tp.fun
            sol_x = res_tp.x.reshape((num_sources, num_dests))

            total_supply = sum(supplies)
            total_demand = sum(demands)

            st.success("計算完成！已依據線性規劃求解器求得全域最佳成本調度方案。")
            m1, m2, m3 = st.columns(3)
            m1.metric("最低總運輸成本 (Z*)", f"${int(round(total_cost)):,} 元")
            m2.metric("總供應總量", f"{fmt_num(total_supply)} 單位")
            m3.metric("總需求總量", f"{fmt_num(total_demand)} 單位")

            st.subheader("Step 7: 最佳調度數量明細表")
            records = []
            for i in range(num_sources):
                for j in range(num_dests):
                    qty = sol_x[i][j]
                    if qty > 1e-3:
                        unit_c = costs_matrix[i][j]
                        subtotal = qty * unit_c
                        records.append({
                            "供應來源地": f"供應地 S{i+1}",
                            "配送目的地": f"需求地 D{j+1}",
                            "每單位運費": f"${fmt_num(unit_c)}",
                            "最佳調度數量": f"{fmt_num(qty)} 單位",
                            "運費小計": f"${fmt_num(subtotal)} 元"
                        })
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

            st.subheader("Step 8: 供需限制式與結果深度分析")
            a_c1, a_c2 = st.columns(2)
            with a_c1:
                st.markdown("**各供給地剩餘容量分析：**")
                for i in range(num_sources):
                    used = sum(sol_x[i, :])
                    rem = supplies[i] - used
                    status = "Binding (已達容量上限)" if rem < 1e-3 else f"Nonbinding (尚餘 {fmt_num(rem)} 單位)"
                    st.write(f"• 供應地 S{i+1}: 供應 {fmt_num(supplies[i])} | 實際調出 {fmt_num(used)} | 狀態: {status}")
            with a_c2:
                st.markdown("**各需求地滿意度分析：**")
                for j in range(num_dests):
                    recv = sum(sol_x[:, j])
                    st.write(f"• 需求地 D{j+1}: 目標需求 {fmt_num(demands[j])} | 實際送達 {fmt_num(recv)} (100% 達成)")

            st.subheader("配送流量配置熱圖 (Flow Heatmap)")
            fig_heat = go.Figure(data=go.Heatmap(
                z=sol_x,
                x=[f"需求地 D{j+1}" for j in range(num_dests)],
                y=[f"供應地 S{i+1}" for i in range(num_sources)],
                colorscale='Blues',
                text=[[f"{fmt_num(sol_x[i][j])}" for j in range(num_dests)] for i in range(num_sources)],
                texttemplate="%{text}",
                textfont={"size": 14}
            ))
            fig_heat.update_layout(height=350)
            st.plotly_chart(fig_heat, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# 頁面三：進階範本二 - 薪資最佳化決策支援系統 (支援自訂 Max Z 權重)
# -------------------------------------------------------------------------------------------------
elif app_mode == "薪資最佳化決策支援系統":
    st.title("薪資最佳化決策支援系統 (DSS)")
    st.caption("進階管理科學問題範本：個人財務與多目標資源分配模型")

    st.sidebar.subheader("模型輸入參數")
    salary = st.sidebar.slider("每月實領總薪資 (元)", min_value=30000, max_value=120000, value=50000, step=1000)
    fixed_cost = st.sidebar.slider("每月固定支出 (元)", min_value=5000, max_value=40000, value=12000, step=500)
    target_gap = st.sidebar.number_input("目標緊急預備金總額 (元)", value=300000, step=10000)
    current_gap = st.sidebar.number_input("目前已存預備金 (元)", value=280000, step=5000)

    persona = st.sidebar.selectbox("偏好情境 (目標函數權重)", ["均衡穩健型", "資產衝刺型", "生活品質型", "自訂權重模式"])

    B = max(0, salary - fixed_cost)
    gap = max(0, target_gap - current_gap)

    weights_map = {
        "均衡穩健型": [1.0, 1.3, 1.3, 2.5, 3.0, 1.6, 1.4],
        "資產衝刺型": [1.0, 1.0, 2.2, 3.0, 2.0, 1.5, 1.0],
        "生活品質型": [1.2, 2.0, 1.2, 1.5, 1.5, 1.2, 2.5]
    }

    var_names = ["基本生活 (x1)", "社交休閒 (x2)", "自我進修 (x3)", "長期投資 (x4)", "緊急預備金 (x5)", "固定儲蓄 (x6)", "個人嗜好 (x7)"]

    if persona == "自訂權重模式":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**請自訂目標函數各項目權重 (w₁ ~ w₇)：**")
        custom_w = []
        default_custom = [1.0, 1.5, 1.5, 2.5, 2.0, 1.5, 1.0]
        for i, name in enumerate(var_names):
            w_val = st.sidebar.slider(
                f"{name} 權重",
                min_value=0.1, max_value=5.0,
                value=default_custom[i], step=0.1,
                key=f"custom_w_{i}"
            )
            custom_w.append(w_val)
        active_weights = custom_w
    else:
        active_weights = weights_map[persona]

    w_formula = " + ".join([f"{w:.1f}x_{{{i+1}}}" for i, w in enumerate(active_weights)])
    st.markdown(f"**當前套用之目標函數：** $\\max Z = {w_formula}$")

    def solve_salary_dss(budget, reserve_gap, w):
        c_obj = [-val for val in w]
        A = [
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, -1, -1, 0, 0, 1],
            [0, -1, 0, 0.2, 0, 0, -1],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0]
        ]
        b = [
            budget, 0.65 * budget, reserve_gap, 0, 0,
            0.15 * budget, -0.07 * budget, -9000, -10000,
            -0.05 * budget, -0.10 * budget
        ]
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(0, None)]*7, method='highs')
        if res.success:
            return True, res.x, -res.fun
        return False, [], 0

    success, sol, max_z = solve_salary_dss(B, gap, active_weights)

    st.subheader("求解結果與目標函數值")
    if not success:
        st.error("求解狀態：無可行解 (Infeasible)。原因：最低生活與投資底限總和超出可支配預算。")
    else:
        asset_sum = sol[3] + sol[4] + sol[5]
        c1, c2, c3 = st.columns(3)
        c1.metric("可支配預算 (B)", f"{int(B):,} 元", "實領薪資扣除固定支出")
        c2.metric("最大目標函數值 (Max Z)", f"{int(round(max_z)):,}", f"情境：{persona}")
        c3.metric("資產累積總額 (x4+x5+x6)", f"{int(round(asset_sum)):,} 元", f"佔可支配預算 {asset_sum/B*100:.1f}%")

        st.success("求解狀態：成功找到最佳解 (Optimal Solution Found)。所有條件皆已達成。")

        st.markdown("---")
        st.subheader("最佳解配置明細")

        col_table, col_chart = st.columns([1, 1])
        with col_table:
            df_optimal = pd.DataFrame({
                "決策變數": ["x1", "x2", "x3", "x4", "x5", "x6", "x7"],
                "分配項目": ["基本生活開銷", "社交/休閒娛樂", "自我進修/學習", "長期投資配置", "緊急預備金提撥", "固定儲蓄提撥", "個人愛好支出"],
                "目標權重": [f"{w:.1f}" for w in active_weights],
                "最佳解金額": [f"{int(round(v)):,} 元" for v in sol],
                "預算佔比": [f"{v/B*100:.1f}%" for v in sol]
            })
            st.dataframe(df_optimal, use_container_width=True, hide_index=True)

        with col_chart:
            fig_pie = go.Figure(data=[go.Pie(
                labels=var_names,
                values=sol,
                hole=0.55,
                textinfo='label+percent',
                marker=dict(colors=['#2563eb', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b'])
            )])
            fig_pie.update_layout(
                title="最佳化支出結構佔比",
                height=300,
                margin=dict(l=10, r=10, t=35, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.subheader("敏感度情境模擬 (What-if Analysis)")
        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sim_salary = st.slider("情境模擬：總薪資 (元)", 30000, 120000, salary + 6000, 1000)
        with sim_c2:
            sim_fixed = st.slider("情境模擬：固定支出 (元)", 5000, 40000, fixed_cost, 500)

        sim_B = max(0, sim_salary - sim_fixed)
        sim_success, sim_sol, sim_z = solve_salary_dss(sim_B, gap, active_weights)

        if sim_success:
            diff_sol = [s - b for s, b in zip(sim_sol, sol)]
            df_compare = pd.DataFrame({
                "指標項目 / 決策變數": var_names + ["總可支配預算 (B)", "最大目標函數值 (Max Z)"],
                "基準方案 (現況)": [f"{int(round(v)):,} 元" for v in sol] + [f"{int(B):,} 元", f"{int(round(max_z)):,}"],
                "模擬情境 (What-if)": [f"{int(round(v)):,} 元" for v in sim_sol] + [f"{int(sim_B):,} 元", f"{int(round(sim_z)):,}"],
                "差異增減 (Δ)": [f"{int(round(v)):+,} 元" for v in diff_sol] + [f"{int(sim_B - B):+,} 元", f"{int(round(sim_z - max_z)):+,}"]
            })
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

    with st.expander("查看數學規劃模型與限制式規格 (Mathematical Model)"):
        st.code("""
決策變數:
x1(生活), x2(社交), x3(進修), x4(投資), x5(預備金), x6(儲蓄), x7(嗜好)

模型常數與外生參數說明:
• B (可支配預算): 實領薪資扣除固定必要支出後之剩餘金額 (B = 總薪資 - 固定支出)
• gap (預備金缺口): 目標預備金與目前已存水位之差額 (gap = max(0, 目標預備金 - 已存預備金))
• wi: 各項決策變數在目標函數中的效用加權權重 (可由使用者自訂或選取預設情境)

目標函數:
Max Z = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7

限制條件:
(1) 總預算上限: x1 + x2 + x3 + x4 + x5 + x6 + x7 <= B
(2) 資產累積上限: x4 + x5 + x6 <= 0.65 * B
(3) 緊急預備金水位限制: x5 <= gap
(4) 嗜好連動約束: -x3 - x4 + x7 <= 0  (即 x7 <= x3 + x4)
(5) 生活平衡約束: -x2 + 0.2*x4 - x7 <= 0 (即 0.2*x4 <= x2 + x7)
(6) 進修上限: x3 <= 0.15 * B
(7) 進修下限: x3 >= 0.07 * B
(8) 基本生活保障: x1 >= 9000
(9) 核心投資底限: x4 >= 10000
(10) 社交休閒底限: x2 >= 0.05 * B
(11) 儲蓄底限: x6 >= 0.10 * B
(12) 非負約束: xi >= 0, for all i
""", language="text")

# -------------------------------------------------------------------------------------------------
# 頁面四：進階範本三 - 可自訂限制式參數之薪資最佳化決策支援系統
# -------------------------------------------------------------------------------------------------
else:
    st.title("可自訂限制式參數之薪資決策系統 (Customizable Constraints DSS)")
    st.caption("進階管理科學模型範本：將通用線性規劃計算器動態輸入融入財務資源配置")

    # 1. 預算與外生常數
    st.subheader("1. 財務基礎參數設定")
    p_c1, p_c2, p_c3, p_c4 = st.columns(4)
    with p_c1:
        param_salary = st.number_input("每月實領總薪資 (元)", min_value=20000, max_value=200000, value=50000, step=1000, key="c_sal")
    with p_c2:
        param_fixed = st.number_input("每月固定支出 (元)", min_value=0, max_value=80000, value=12000, step=500, key="c_fix")
    with p_c3:
        param_tg = st.number_input("目標緊急預備金總額 (元)", value=300000, step=10000, key="c_tg")
    with p_c4:
        param_cg = st.number_input("目前已存預備金 (元)", value=280000, step=5000, key="c_cg")

    B_val = max(0.0, float(param_salary - param_fixed))
    gap_val = max(0.0, float(param_tg - param_cg))
    st.info(f"💡 目前計算之 **可支配預算 (B)** 為 **{int(B_val):,} 元**；**緊急預備金缺口 (gap)** 為 **{int(gap_val):,} 元**。")

    # 2. 限制式參數自訂調整區
    st.markdown("---")
    st.subheader("2. 限制式邊界參數自訂調整")
    st.caption("使用者可依個人財務偏好與需求，彈性自訂各項不等式之常數與比例門檻。")

    con_col1, con_col2 = st.columns(2)
    with con_col1:
        st.markdown("**金額型下限約束 (保障門檻)：**")
        param_x1_min = st.number_input("基本生活保障下限 (x₁ ≥ ? 元)", value=9000.0, step=500.0, key="par_x1_min")
        param_x4_min = st.number_input("核心長期投資底限 (x₄ ≥ ? 元)", value=10000.0, step=1000.0, key="par_x4_min")
        
        st.markdown("**連動關係係數：**")
        param_balance_ratio = st.number_input("投資連動生活平衡係數 (即 係數*x₄ - x₂ - x₇ ≤ 0)", value=0.20, step=0.05, min_value=0.0, max_value=1.0, key="par_bal_ratio")

    with con_col2:
        st.markdown("**比例型約束 (佔可支配預算 B 之比例 %)：**")
        param_asset_ratio = st.slider("資產累積上限比例 (x₄ + x₅ + x₆ ≤ % × B)", min_value=30, max_value=90, value=65, step=5, key="par_asset") / 100.0
        param_x3_max_ratio = st.slider("自我進修上限比例 (x₃ ≤ % × B)", min_value=5, max_value=40, value=15, step=1, key="par_x3_max") / 100.0
        param_x3_min_ratio = st.slider("自我進修下限比例 (x₃ ≥ % × B)", min_value=0, max_value=20, value=7, step=1, key="par_x3_min") / 100.0
        param_x2_min_ratio = st.slider("社交休閒保障比例 (x₂ ≥ % × B)", min_value=0, max_value=20, value=5, step=1, key="par_x2_min") / 100.0
        param_x6_min_ratio = st.slider("固定儲蓄提撥比例 (x₆ ≥ % × B)", min_value=0, max_value=30, value=10, step=1, key="par_x6_min") / 100.0

    # 3. 目標函數權重設定
    st.markdown("---")
    st.subheader("3. 目標函數各項目權重設定")
    var_list_3 = ["基本生活 (x1)", "社交休閒 (x2)", "自我進修 (x3)", "長期投資 (x4)", "緊急預備金 (x5)", "固定儲蓄 (x6)", "個人嗜好 (x7)"]
    w_cols = st.columns(7)
    weights_3 = []
    default_w3 = [1.0, 1.3, 1.3, 2.5, 3.0, 1.6, 1.4]
    for i, name in enumerate(var_list_3):
        with w_cols[i]:
            w_i = st.number_input(f"x{i+1} 權重", value=default_w3[i], step=0.1, key=f"w3_{i}")
            weights_3.append(w_i)

    # 4. 動態建構限制式結構清單
    constraints_custom = [
        {"name": "總預算上限", "a": [1, 1, 1, 1, 1, 1, 1], "op": "<=", "rhs": B_val},
        {"name": "資產累積上限", "a": [0, 0, 0, 1, 1, 1, 0], "op": "<=", "rhs": param_asset_ratio * B_val},
        {"name": "緊急預備金水位", "a": [0, 0, 0, 0, 1, 0, 0], "op": "<=", "rhs": gap_val},
        {"name": "嗜好連動約束", "a": [0, 0, -1, -1, 0, 0, 1], "op": "<=", "rhs": 0.0},
        {"name": "生活平衡約束", "a": [0, -1, 0, param_balance_ratio, 0, 0, -1], "op": "<=", "rhs": 0.0},
        {"name": "進修上限", "a": [0, 0, 1, 0, 0, 0, 0], "op": "<=", "rhs": param_x3_max_ratio * B_val},
        {"name": "進修下限", "a": [0, 0, 1, 0, 0, 0, 0], "op": ">=", "rhs": param_x3_min_ratio * B_val},
        {"name": "基本生活保障", "a": [1, 0, 0, 0, 0, 0, 0], "op": ">=", "rhs": param_x1_min},
        {"name": "核心投資底限", "a": [0, 0, 0, 1, 0, 0, 0], "op": ">=", "rhs": param_x4_min},
        {"name": "社交休閒底限", "a": [0, 1, 0, 0, 0, 0, 0], "op": ">=", "rhs": param_x2_min_ratio * B_val},
        {"name": "固定儲蓄底限", "a": [0, 0, 0, 0, 0, 1, 0], "op": ">=", "rhs": param_x6_min_ratio * B_val}
    ]

    # 即時顯示數學模型預覽 (純數學表示法，修正 1 與 -1 顯示)
    st.markdown("---")
    st.subheader("4. 動態產生之數學模型規格")
    with st.expander("點擊展開查看依使用者自訂參數動態生成之 LP 模型", expanded=True):
        obj_str3 = " + ".join([f"{fmt_num(weights_3[i])}x_{{{i+1}}}" for i in range(7)])
        latex_lines3 = [f"\\begin{{aligned}} \\max\\quad Z &= {obj_str3} \\\\\n\\text{{subject to}}\\quad "]
        for con in constraints_custom:
            nonzero_terms = []
            for j in range(7):
                coef = con['a'][j]
                if coef == 0:
                    continue
                # 精準處理係數 1 與 -1 的格式
                if abs(coef - 1.0) < 1e-5:
                    nonzero_terms.append(f"x_{{{j+1}}}")
                elif abs(coef - (-1.0)) < 1e-5:
                    nonzero_terms.append(f"-x_{{{j+1}}}")
                else:
                    nonzero_terms.append(f"{fmt_num(coef)}x_{{{j+1}}}")

            lhs_math = " + ".join(nonzero_terms).replace("+ -", "- ")
            sym = "\\le" if con['op'] == '<=' else ("\\ge" if con['op'] == '>=' else "=")
            latex_lines3.append(f"{lhs_math} &{sym} {fmt_num(con['rhs'])}")
        latex_lines3.append("x_i &\\ge 0, \\quad \\forall i=1,\\dots,7")
        latex_preview3 = " \\\\\n".join(latex_lines3) + "\n\\end{aligned}"
        st.latex(latex_preview3)

    # 5. 求解與結果呈現
    if st.button("依自訂限制式參數求解 (Solve Custom LP)", type="primary"):
        success_3, sol_data_3, err_msg_3 = solve_lp_highs("Max", weights_3, constraints_custom, non_neg=True)

        if not success_3:
            st.error(f"求解失敗：在目前的參數設定下無可行解 (Infeasible)！原因：底限與固定下限總額超出可支配預算 ({err_msg_3})。請調降最低保障金額或提高薪資。")
        else:
            sol_3 = sol_data_3['x']
            best_z_3 = sol_data_3['z']
            sp_3 = sol_data_3['shadow_prices']

            st.success("計算成功！已依據您自訂的限制條件求得最佳財務資源配置方案。")

            c_res1, c_res2, c_res3 = st.columns(3)
            with c_res1:
                st.metric("可支配預算 (B)", f"{int(B_val):,} 元")
            with c_res2:
                st.metric("最佳目標函數值 (Max Z*)", fmt_num(best_z_3))
            with c_res3:
                asset_tot_3 = sol_3[3] + sol_3[4] + sol_3[5]
                st.metric("資產累積總額 (x₄+x₅+x₆)", f"{int(round(asset_tot_3)):,} 元", f"佔預算 {asset_tot_3/B_val*100:.1f}%")

            st.markdown("---")
            st.subheader("最佳資金配置明細與佔比")
            col_t3, col_p3 = st.columns([1, 1])
            with col_t3:
                df_custom_opt = pd.DataFrame({
                    "決策變數": [f"x{j+1}" for j in range(7)],
                    "項目名稱": ["基本生活開銷", "社交/休閒娛樂", "自我進修/學習", "長期投資配置", "緊急預備金提撥", "固定儲蓄提撥", "個人嗜好支出"],
                    "配置金額": [f"{int(round(val)):,} 元" for val in sol_3],
                    "佔預算比例": [f"{val/B_val*100:.1f}%" if B_val > 0 else "0%" for val in sol_3]
                })
                st.dataframe(df_custom_opt, use_container_width=True, hide_index=True)

            with col_p3:
                fig_pie3 = go.Figure(data=[go.Pie(
                    labels=var_list_3,
                    values=sol_3,
                    hole=0.55,
                    textinfo='label+percent',
                    marker=dict(colors=['#2563eb', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b'])
                )])
                fig_pie3.update_layout(
                    title="自訂參數最佳支出結構佔比",
                    height=300,
                    margin=dict(l=10, r=10, t=35, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_pie3, use_container_width=True)

            st.subheader("限制式與敏感度分析")
            table_custom_data = []
            for k, con in enumerate(constraints_custom):
                lhs_val = sum(con['a'][j] * sol_3[j] for j in range(7))
                diff = abs(lhs_val - con['rhs'])
                is_binding = diff < 1e-4
                slack = 0.0 if is_binding else abs(con['rhs'] - lhs_val)
                sp_val = sp_3[k]

                table_custom_data.append({
                    "限制式名稱": con['name'],
                    "左側計算值 (LHS)": fmt_num(lhs_val),
                    "符號": con['op'],
                    "右側常數 (RHS)": fmt_num(con['rhs']),
                    "鬆弛量 (Slack)": fmt_num(slack),
                    "狀態判定": "Binding (緊束瓶頸)" if is_binding else "Nonbinding (有餘量)",
                    "影子價格 (Shadow Price)": fmt_num(sp_val)
                })

            st.dataframe(pd.DataFrame(table_custom_data), use_container_width=True, hide_index=True)

    # 範本三最底部的數學規劃模型規格與參數說明 (比照範本二)
    with st.expander("查看數學規劃模型與限制式規格"):
        st.code("""
決策變數:
x1(生活), x2(社交), x3(進修), x4(投資), x5(預備金), x6(儲蓄), x7(嗜好)

模型常數與外生參數說明:
• B (可支配預算): 實領薪資扣除固定必要支出後之剩餘金額 (B = 總薪資 - 固定支出)
• gap (預備金缺口): 目標預備金與目前已存水位之差額 (gap = max(0, 目標預備金 - 已存預備金))
• wi: 各項決策變數在目標函數中的效用加權權重 (可由使用者自訂調整)

目標函數:
Max Z = w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7

限制條件 (本範本支援使用者彈性自訂參數):
(1) 總預算上限: x1 + x2 + x3 + x4 + x5 + x6 + x7 <= B
(2) 資產累積上限: x4 + x5 + x6 <= [資產累積上限比例] * B
(3) 緊急預備金水位限制: x5 <= gap
(4) 嗜好連動約束: -x3 - x4 + x7 <= 0  (即 x7 <= x3 + x4)
(5) 生活平衡約束: -x2 + [連動係數]*x4 - x7 <= 0
(6) 進修上限: x3 <= [進修上限比例] * B
(7) 進修下限: x3 >= [進修下限比例] * B
(8) 基本生活保障: x1 >= [基本生活保障金額]
(9) 核心投資底限: x4 >= [核心長期投資底限]
(10) 社交休閒底限: x2 >= [社交休閒保障比例] * B
(11) 儲蓄底限: x6 >= [固定儲蓄提撥比例] * B
(12) 非負約束: xi >= 0, for all i=1,...,7
""", language="text")
