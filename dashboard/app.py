from __future__ import annotations
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1]
AGENT_DIR=ROOT/"Phase5_Agentic_AI"
if str(AGENT_DIR) not in sys.path: sys.path.insert(0,str(AGENT_DIR))
from agents.agentic_controller import run_agentic_case
from tools.raw_mat_loader import load_bearing_mat
from tools.raw_signal_validator import validate_signal
from tools.raw_feature_pipeline import extract_case_features
from tools.operator_status_tool import operator_status_tool

st.set_page_config(
    page_title="Bearing Diagnostic Copilot",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Overall page background */
.stApp {
    background: linear-gradient(135deg, #eef8f7 0%, #eaf4fb 100%);
}

/* Main content */
.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    padding: 24px 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f766e, #2563eb);
    color: white;
    margin-bottom: 18px;
}

.hero h1 {
    margin: 0;
    font-size: 2rem;
}

.hero p {
    margin: .4rem 0 0;
    opacity: .9;
}

/* Cards */
.card {
    padding: 18px 20px;
    border: 1px solid #d7e7e8;
    border-radius: 16px;
    background: rgba(255,255,255,.78);
    min-height: 112px;
    box-shadow: 0 2px 8px rgba(15,118,110,.06);
}

.label {
    font-size: .78rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    opacity: .65;
    font-weight: 700;
}

.value {
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 6px;
}

/* Status card */
.status-card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #d7e7e8;
    background: rgba(255,255,255,.82);
    margin: 10px 0 18px;
    box-shadow: 0 3px 10px rgba(15,118,110,.06);
}

.status-name {
    font-size: 2rem;
    font-weight: 900;
}

.diagnosis {
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 5px;
}

/* Information notice */
.notice {
    padding: 12px 15px;
    border-radius: 12px;
    background: #f0fdfa;
    border: 1px solid #b7e4df;
}

/* Small text */
.small {
    font-size: .86rem;
    opacity: .7;
}

/* Agent path */
.path {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    word-break: break-word;
}

/* Streamlit metrics */
[data-testid="stMetricValue"] {
    font-weight: 800;
}

/* Resolution / summary metrics: allow long human-readable labels to wrap. */
.summary-card {
    min-height: 96px;
    padding: 4px 0 8px;
}
.summary-card .summary-label {
    font-size: .78rem;
    color: #334155;
    margin-bottom: 5px;
}
.summary-card .summary-value {
    font-size: 1.55rem;
    line-height: 1.18;
    font-weight: 800;
    color: #172554;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: normal;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f4fbfa;
    border-right: 1px solid #d7e7e8;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #d7e7e8;
    border-radius: 14px;
    background: rgba(255,255,255,.7);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.65);
    border-radius: 14px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>⚙️ Bearing Diagnostic Copilot</h1><p>Evidence-grounded agentic decision support for industrial bearing condition assessment</p><p class="small">• Offline .mat validation / Research prototype &emsp; • Decision support only &emsp; • Not direct plant control</p></div>',unsafe_allow_html=True)

with st.sidebar:
    st.header("Run a diagnosis")
    uploaded=st.file_uploader("MATLAB vibration recording",type=["mat"],help="Upload one raw bearing vibration .mat recording.")
    model=st.text_input("Local LLM model",value=os.getenv("OLLAMA_MODEL","qwen3:4b"),help="Latency-first local profile for 16 GB RAM / 4 GB GPU: qwen3:4b. qwen3:8b and qwen3:14b remain supported for quality benchmarking.")
    st.caption("Recommended local profile: **Qwen3 4B + concise structured inference**. Use Qwen3 8B/14B for quality benchmarking when higher latency is acceptable.")
    max_steps=st.slider("Investigation budget (steps)",8,25,15,help="Maximum planning/tool actions. Terminal completion does not require a synthetic final step.")
    run=st.button("▶  Run diagnosis",type="primary",use_container_width=True)
    st.divider()
    st.caption("Safety policy")
    st.caption("Qwen3 provides diagnostic reasoning; deterministic tools provide measured evidence; the safety guard constrains actions. No direct machine control.")

if uploaded is None:
    st.info("Upload a raw `.mat` recording in the sidebar to begin.")
    st.markdown('<div class="notice"><b>Agent workflow</b><br><span class="path">RAW SIGNAL → EVIDENCE → LLM INVESTIGATION → NEW EVIDENCE ↺ → LLM DIAGNOSIS → ROOT CAUSE → OPERATOR ACTION</span></div>',unsafe_allow_html=True)
    st.stop()

key=(uploaded.name,uploaded.size)
if st.session_state.get("upload_key")!=key:
    st.session_state["upload_key"]=key
    st.session_state.pop("state",None)
    st.session_state["run_summary"]=None

mat_path=None
try:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".mat") as tmp:
        tmp.write(uploaded.getbuffer()); mat_path=tmp.name
    raw=load_bearing_mat(mat_path,validate=True)
    validation=validate_signal(raw["signal"],raw["sampling_rate"])
except Exception as exc:
    st.error(f"Input could not be loaded: {type(exc).__name__}: {exc}"); st.stop()
finally:
    pass

if validation.get("status")!="valid":
    st.error("Diagnosis blocked: signal validation failed.")
    with st.expander("Validation details"): st.json(validation)
    st.stop()

sig=np.asarray(raw["signal"],dtype=float).reshape(-1); fs=float(raw["sampling_rate"])
try: available_windows=len(extract_case_features(mat_path))
except Exception: available_windows=0

c=st.columns(5)
for col,label,value in zip(c,["Samples","Sampling rate","Duration","Available windows","Input"],[f"{len(sig):,}",f"{fs:,.0f} Hz",f"{len(sig)/fs:.2f} s",str(available_windows),"VALID"]):
    with col: st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>',unsafe_allow_html=True)
st.write("")
st.caption(f"Loaded recording: **{uploaded.name}** • {uploaded.size/1024/1024:.1f} MB")

# ---------------------------------------------------------------------------
# RAW SIGNAL PREVIEW
# ---------------------------------------------------------------------------
# Show the actual uploaded vibration signal immediately after loading. For
# large recordings, an extrema-preserving display envelope keeps the chart
# responsive without hiding important peaks. The values remain raw dataset
# amplitude units; no diagnosis is inferred from this visualization.
def _raw_time_series_for_display(signal: np.ndarray, sampling_rate: float, max_points: int = 20000):
    x = np.asarray(signal, dtype=float).reshape(-1)
    n = x.size
    if n <= max_points:
        idx = np.arange(n, dtype=int)
        return idx / float(sampling_rate), x, False

    # Preserve local minima/maxima so transient peaks are not silently lost.
    bins = max(1, max_points // 2)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    out_idx = []
    out_val = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        segment = x[start:end]
        local_min = start + int(np.argmin(segment))
        local_max = start + int(np.argmax(segment))
        if local_min <= local_max:
            out_idx.extend((local_min, local_max))
            out_val.extend((x[local_min], x[local_max]))
        else:
            out_idx.extend((local_max, local_min))
            out_val.extend((x[local_max], x[local_min]))
    idx = np.asarray(out_idx, dtype=int)
    vals = np.asarray(out_val, dtype=float)
    return idx / float(sampling_rate), vals, True

with st.expander("📈 Raw vibration signal — amplitude vs time", expanded=True):
    plot_t, plot_x, is_reduced = _raw_time_series_for_display(sig, fs)
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scattergl(
        x=plot_t, y=plot_x, mode="lines", name="Vibration amplitude",
        line=dict(width=1)
    ))
    fig_raw.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=55, b=55),
        title=dict(text="Raw vibration waveform", x=0.02),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (raw dataset units)",
        hovermode="x unified",
        showlegend=False,
    )
    fig_raw.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_raw, use_container_width=True, config={"displaylogo": False, "responsive": True})
    if is_reduced:
        st.caption("Large recording: the chart uses an extrema-preserving display envelope for responsiveness; the uploaded raw signal is unchanged.")
    else:
        st.caption("Showing the uploaded raw samples directly. Amplitude units follow the MATLAB dataset; no physical unit is assumed.")

if run:
    with st.spinner("Running diagnostic tools and agentic investigation…"):
        state=run_agentic_case(f"CASE_{Path(uploaded.name).stem}",{"mat_file":mat_path},model=model,max_steps=max_steps)
        st.session_state["state"]=state
        st.session_state["run_summary"]={"file":uploaded.name,"status":state.status,"diagnosis":state.final_diagnosis,"run_id":state.run_id}

state=st.session_state.get("state")
if state is None:
    st.info("Input is valid. Click **Run diagnosis** to start the investigation.")
    st.stop()

status=operator_status_tool(state)
status_map={"HEALTHY":"🟢","NO CONFIRMED FAULT":"🟢","WARNING":"🟡","FAULT":"🟠","CRITICAL":"🔴"}
icon=status_map.get(status["status"],"⚪")

def _display_diagnosis(value: str) -> str:
    labels={
        "NO_CONFIRMED_FAULT":"No confirmed bearing fault",
        "InnerRaceFault":"Inner Race Fault",
        "OuterRaceFault":"Outer Race Fault",
        "Healthy":"Healthy",
        "Healthy_or_Insufficient":"No confirmed fault",
        "UNRESOLVED":"Unresolved",
    }
    return labels.get(str(value), str(value).replace("_", " "))

def _display_resolution(value: str) -> str:
    labels={
        "NO_CONFIRMED_FAULT":"No confirmation",
        "RESOLVED_BY_STRONG_PHYSICS":"Strong physics",
        "CONFIRMED_BY_STRONG_PHYSICS":"Strong physics",
        "CONFIRMED_BY_INDEPENDENT_AGREEMENT":"Independent agreement",
        "CONFIRMED_BY_MULTI_WINDOW_AGREEMENT":"Multi-source agreement",
        "STRONG_AGREEMENT":"Strong agreement",
    }
    return labels.get(str(value), "Not established")

display_diagnosis=_display_diagnosis(status.get("diagnosis","UNRESOLVED"))

st.divider()
st.markdown(f'<div class="status-card"><div class="label">Machine status</div><div class="status-name">{icon} {status["status"]}</div><div class="diagnosis">{display_diagnosis}</div><div class="small">Severity / risk: <b>{status["severity"]}</b></div></div>',unsafe_allow_html=True)

m1,m2,m3,m4=st.columns(4)
resolution_basis=_display_resolution(state.final_resolution)
with m1:
    st.markdown(f'<div class="summary-card"><div class="summary-label">Resolution basis</div><div class="summary-value">{resolution_basis}</div></div>',unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="summary-card"><div class="summary-label">Investigation steps</div><div class="summary-value">{sum(1 for x in state.tool_history if x.get("counts_toward_budget", True))}</div></div>',unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="summary-card"><div class="summary-label">LLM planning decisions</div><div class="summary-value">{sum(1 for x in state.tool_history if x.get("decision_source")=="ollama")}</div></div>',unsafe_allow_html=True)
with m4:
    runtime_s = (float(getattr(state, "total_duration_ms", 0.0)) / 1000.0)
    runtime_label = f"{runtime_s:.1f} s" if runtime_s > 0 else "N/A"
    st.markdown(f'<div class="summary-card"><div class="summary-label">Runtime</div><div class="summary-value">{runtime_label}</div></div>',unsafe_allow_html=True)
st.caption(state.final_confidence_explanation or "No calibrated case-level probability is reported; source-specific ML probabilities remain available in the evidence view.")

left,right=st.columns([1.2,.8])
with left:
    st.subheader("Why this diagnosis?")
    for x in status["why"]: st.markdown(f"• {x}")
with right:
    st.subheader("Recommended action")
    st.markdown(f'<div class="notice">{status["recommended_action"]}</div>',unsafe_allow_html=True)

st.subheader("Trend & limitations")
st.info(status["trend"]["message"])
st.caption(status["limitation"])

with st.expander("Engineer / Researcher View",expanded=False):
    tabs=st.tabs(["Evidence","LLM Diagnosis","Knowledge Graph","Agent trace","RCA & report","Run metadata"])
    with tabs[0]:
        st.markdown("**Evidence status**")
        eq=state.evidence_quality_result or {}
        qcols=st.columns(5)
        qvals=[("ML quality",eq.get("ml_evidence_quality")),("Physics quality",eq.get("physics_evidence_quality")),("Cross-source",eq.get("cross_source_quality")),("Multi-window",eq.get("multi_window_evidence_quality")),("Overall",eq.get("overall_evidence_quality"))]
        for col,(label,val) in zip(qcols,qvals):
            with col: st.metric(label,f"{float(val):.3f}" if val is not None else "N/A")
        st.caption(f"Temporal trend evidence: {eq.get('temporal_evidence_status','NOT_AVAILABLE')} — multi-window consistency is not a temporal health trend.")
        st.caption("ML probabilities are source-specific. No synthetic case-level probability is reported because this prototype does not include a validated calibration model.")
        st.json({"ML":state.ml_result,"Physics":state.physics_result,"Fusion":state.fusion_result,"EvidenceQuality":state.evidence_quality_result,"AdaptiveInvestigation":state.adaptive_investigation_result,"Stopping":state.evidence_stopping_result,"Reanalysis":state.reanalysis_result})
        p=(state.ml_result or {}).get("probabilities") or {}
        if p:
            fig=go.Figure(go.Bar(x=list(p.keys()),y=[float(v) for v in p.values()]))
            fig.update_layout(height=300,margin=dict(l=20,r=20,t=20,b=60),yaxis_title="Probability",xaxis_title="Class")
            st.plotly_chart(fig,use_container_width=True)
    with tabs[1]:
        llm=state.llm_diagnostic_result or {}
        if llm.get("status")=="success":
            st.success(f"LLM diagnostic judgment: {_display_diagnosis(llm.get('diagnosis','UNRESOLVED'))}")
            a,b,c,d=st.columns(4)
            a.metric("Fault location", str(llm.get("fault_location") or "Not established"))
            b.metric("LLM confidence", str(llm.get("diagnostic_confidence") or "N/A").title())
            c.metric("LLM model", str(llm.get("model") or state.llm_model))
            d.metric("Reasoning source", "Evidence-grounded LLM")
            st.markdown("### Diagnostic reasoning")
            st.info(llm.get("reasoning") or "No reasoning returned.")
            st.markdown("### Observed evidence")
            for item in llm.get("observed_evidence",[]) or []: st.markdown(f"• {item}")
            st.markdown("### Inferred mechanism")
            st.write(llm.get("inferred_mechanism") or "Not established")
            st.markdown("### Possible root causes")
            for item in llm.get("possible_root_causes",[]) or []: st.markdown(f"• {item}")
            st.caption(f"Root-cause status: {llm.get('root_cause_status','N/A')}")
            if llm.get("semantic_corrections"):
                st.caption("Safety/semantic verification corrections applied:")
                for item in llm.get("semantic_corrections", []): st.markdown(f"• {item}")
            st.markdown("### Prioritized operator actions")
            for i,item in enumerate(llm.get("operator_actions",[]) or [],1): st.markdown(f"**{i}.** {item}")
            st.markdown("### Additional investigation")
            st.write(llm.get("additional_investigation") or "No additional investigation requested.")
        else:
            st.warning("The LLM diagnostic judgment was unavailable or rejected; the deterministic evidence chain remains the safety fallback.")
            if llm: st.json(llm)

    with tabs[2]:
        kg=state.knowledge_graph_context or {}
        st.markdown("### Case-aware diagnostic knowledge graph")
        st.caption("The graph links measured evidence to fault hypotheses, mechanisms and operator checks. The current prototype uses a Neo4j-compatible node/relationship schema locally.")
        nodes=kg.get("nodes",[]) or []
        edges=kg.get("relationships",[]) or []
        if nodes:
            positions={}
            import math
            radius=max(1.0, len(nodes)/(2*math.pi))
            for i,n in enumerate(nodes):
                ang=2*math.pi*i/max(1,len(nodes)); positions[n["id"]]=(radius*math.cos(ang),radius*math.sin(ang))
            figkg=go.Figure()
            domain_added=False; case_added=False
            for e in edges:
                if e["source"] in positions and e["target"] in positions:
                    x0,y0=positions[e["source"]]; x1,y1=positions[e["target"]]
                    scope=e.get("scope","domain_knowledge")
                    is_case=(scope=="case_evidence")
                    figkg.add_trace(go.Scatter(
                        x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                        line=dict(width=2 if is_case else 1, dash="solid" if is_case else "dot"),
                        hoverinfo="text", text=[e.get("relation","")]*3,
                        name="Current case evidence" if is_case and not case_added else "Domain knowledge" if (not is_case and not domain_added) else None,
                        showlegend=(is_case and not case_added) or ((not is_case) and not domain_added),
                        legendgroup="case" if is_case else "domain"
                    ))
                    if is_case: case_added=True
                    else: domain_added=True
            figkg.add_trace(go.Scatter(x=[positions[n["id"]][0] for n in nodes],y=[positions[n["id"]][1] for n in nodes],mode="markers+text",text=[n["label"] for n in nodes],textposition="top center",marker=dict(size=18),customdata=[n["type"] for n in nodes],hovertemplate="%{text}<br>Type: %{customdata}<extra></extra>",showlegend=False))
            figkg.update_layout(height=620,margin=dict(l=10,r=10,t=20,b=10),xaxis=dict(visible=False),yaxis=dict(visible=False),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",legend=dict(orientation="h",y=1.02,x=0))
            st.plotly_chart(figkg,use_container_width=True,config={"displaylogo":False})
            st.caption(f"{len(nodes)} graph nodes • {len(edges)} relationships")
            with st.expander("Graph relationships"): st.json(edges)

    with tabs[3]:
        st.markdown(f'**Investigation path:** <span class="path">{" → ".join(x.get("action","") for x in state.tool_history)}</span>',unsafe_allow_html=True)
        for item in state.tool_history:
            src=item.get("decision_source","unknown")
            badge=("🧠 LLM" if src=="ollama" else "🛡️ POLICY" if src=="policy_required" else "📋 FINALIZATION" if src=="terminal_required" else "⚠️ SAFETY")
            step_label = "Terminal finalization" if item.get("counts_toward_budget") is False else f"Step {item.get('step')}"
            st.markdown(f"**{step_label} — `{item.get('action')}`** · {badge}")
            duration = item.get("duration_ms")
            timing = f" • {float(duration)/1000.0:.2f}s" if duration is not None else ""
            st.caption(f"{item.get('reason','')}{timing}")
        st.caption(f"LLM decisions: {sum(1 for x in state.tool_history if x.get('decision_source')=='ollama')} • policy-required: {sum(1 for x in state.tool_history if x.get('decision_source')=='policy_required')} • safety/fallback: {sum(1 for x in state.tool_history if x.get('decision_source') in {'safety_fallback','ollama_rationale_replaced'})}")
    with tabs[4]:
        if state.rca_result: st.json(state.rca_result)
        st.markdown("**Operator report**")
        st.text((state.operator_report or {}).get("report", "Report unavailable."))
    with tabs[5]:
        st.json({"run_id":state.run_id,"case_id":state.case_id,"llm_model":state.llm_model,"llm_temperature":state.llm_temperature,"max_steps":state.max_steps,"started_at":state.started_at,"completed_at":state.completed_at,"total_duration_ms":getattr(state,"total_duration_ms",0.0),"status":state.status,"final_resolution":state.final_resolution,"final_diagnosis":state.final_diagnosis,"final_confidence":state.final_confidence,"final_confidence_type":state.final_confidence_type,"final_confidence_explanation":state.final_confidence_explanation})

export={"status":state.status,"operator_status":status,"final_diagnosis":state.final_diagnosis,"final_confidence":state.final_confidence,"final_confidence_type":state.final_confidence_type,"final_confidence_explanation":state.final_confidence_explanation,"final_resolution":state.final_resolution,"tool_history":state.tool_history,"run_id":state.run_id,"case_id":state.case_id,"llm_model":state.llm_model,"llm_temperature":state.llm_temperature,"max_steps":state.max_steps,"started_at":state.started_at,"completed_at":state.completed_at,"total_duration_ms":getattr(state,"total_duration_ms",0.0),"llm_diagnostic":state.llm_diagnostic_result,"knowledge_graph":state.knowledge_graph_context}
st.download_button("⬇ Export diagnostic JSON",json.dumps(export,default=str,indent=2),file_name=f"{state.case_id}_diagnosis.json",mime="application/json",use_container_width=False)
