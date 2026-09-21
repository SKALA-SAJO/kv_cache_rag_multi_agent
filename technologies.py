"""비교 대상 기술 정의 (RAG-Design PDF A.2, A.3 절).

Human 기반 기술 선정 방식으로 이미 확정된 두 기술의 메타데이터다.
Graph의 '기술 선정 Node'가 이 값을 selected_technologies State로 주입한다.
"""

SELECTED_TECHNOLOGIES = {
    "DeepSeek-V2 MLA": {
        "category": "SW",
        "core_approach": (
            "Multi-head Latent Attention(MLA)으로 Key/Value를 저차원 latent "
            "representation으로 공동 압축하여 KV Cache 저장량을 줄임"
        ),
        "source_document": "deepseek_v2_mla.pdf",
        "paper_reference": "DeepSeek-AI (2024). DeepSeek-V2. arXiv:2405.04434.",
    },
    "InfiniGen": {
        "category": "HW·인프라",
        "core_approach": (
            "호스트(CPU) 메모리에 KV Cache를 두고, 필요한 항목만 예측하여 GPU로 "
            "선택적으로 프리페치하는 동적 오프로딩 기반 서빙 시스템"
        ),
        "source_document": "infinigen.pdf",
        "paper_reference": (
            "Lee, W. et al. (2024). InfiniGen. USENIX OSDI 2024. arXiv:2406.19707."
        ),
    },
}
