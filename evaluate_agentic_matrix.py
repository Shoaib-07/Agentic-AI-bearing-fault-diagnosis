"""Agentic case matrix using the real controller and an optional deterministic mock planner.
Use --real-llm on a machine with Ollama; default mock mode validates controller/tool/state behavior.
"""
from pathlib import Path
import argparse, json, os, re, sys
import pandas as pd
ROOT=Path(__file__).resolve().parent; AGENT=ROOT/"Phase5_Agentic_AI"; sys.path.insert(0,str(AGENT))
import agents.agentic_controller as controller
CASES=[("Healthy","baseline_3.mat","test_data"),("Inner Race","InnerRaceFault_vload_6.mat","test_data"),("Outer Race","OuterRaceFault_3.mat","test_data"),("Conflict","OuterRaceFault_vload_7.mat","test_data"),("Low-confidence + strong physics","OuterRaceFault_vload_5.mat","train_data")]
def mock_chat(model,messages,format=None,options=None):
    prompt=messages[-1]["content"]
    m=re.search(r"ELIGIBLE ACTIONS .*?:\n(\{.*?\})\n\nDECISION RULES",prompt,re.S); eligible=list(json.loads(m.group(1)))
    # Deterministic evidence-sensitive policy used only when Ollama is unavailable.
    action=eligible[0]
    if "evidence_quality" in eligible and "reanalyze_signal" in eligible:
        summary=prompt.lower()
        # For validation only: choose quality when initial ML/physics evidence agrees;
        # choose new signal evidence when the state indicates conflict/uncertainty.
        if '"conflict": true' in summary or '"conflict_detected": true' in summary or '"confidence": 0.' in summary:
            action="reanalyze_signal"
        else:
            action="evidence_quality"
    return {"message":{"content":json.dumps({"action":action,"reason":"Selected from the currently eligible actions using the supplied evidence state."})}}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--real-llm",action="store_true"); args=ap.parse_args()
    if not args.real_llm: controller.chat=mock_chat
    rows=[]
    for label,name,folder in CASES:
        path=ROOT/folder/name; s=controller.run_agentic_case("MATRIX_"+path.stem,{"mat_file":str(path)},model=os.getenv("OLLAMA_MODEL","qwen3:4b"),max_steps=20)
        acts=[x.get("action") for x in s.tool_history]
        rows.append({"case":label,"file":name,"ML":(s.ml_result or {}).get("fault"),"Physics":(s.physics_result or {}).get("diagnosis"),"Conflict":(s.reanalysis_result or {}).get("conflict_detected") if s.reanalysis_result else ((s.ml_result or {}).get("fault") != (s.physics_result or {}).get("diagnosis") if s.ml_result and s.physics_result else None),"Agent Actions":" → ".join(acts),"Final Diagnosis":s.final_diagnosis,"Steps":len(acts),"LLM Decisions":sum(x.get("decision_source")=="ollama" for x in s.tool_history),"Policy Decisions":sum(x.get("decision_source")=="policy_required" for x in s.tool_history),"Safety/Fallbacks":sum(x.get("decision_source") in {"safety_fallback","ollama_rationale_replaced"} for x in s.tool_history),"Decision Sources":", ".join(sorted(set(x.get("decision_source","") for x in s.tool_history)) )})
    df=pd.DataFrame(rows); out=ROOT/"research_evaluation_results"; out.mkdir(exist_ok=True); df.to_csv(out/"agentic_case_comparison.csv",index=False); print(df.to_string(index=False)); print("\nSaved:",out/"agentic_case_comparison.csv")
if __name__=="__main__": main()
