with open('docs/evaluation_result/Comprehensive_Final_Evaluation_Report.md', 'r') as f:
    content = f.read()

t10_old = """| Variant | Classified Sentences | Type: Retrieval | Type: Model | Type: Response | Type: Query |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LettuceDetect** | 316 | 9 | 293 | 14 | - |
| **full_verifier_filter** | 658 | 572 | 39 | 45 | 2 |
| verifier_nli_filter | 670 | 578 | 44 | 48 | - |"""

t10_new = """| Variant | Classified Sentences | Type: Retrieval | Type: Model | Type: Response | Type: Query |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LettuceDetect** | 316 | 9 | 293 | 14 | - |
| **full_verifier_filter** | 658 | 572 | 39 | 45 | 2 |
| verifier_grounded_filter | 677 | 579 | 45 | 53 | - |
| verifier_intrinsic_filter | 689 | 604 | 37 | 48 | - |
| **verifier_nli_filter** | 670 | 578 | 44 | 48 | - |
| verifier_self_agreement_filter | 691 | 596 | 31 | 64 | - |"""

content = content.replace(t10_old, t10_new)

t11_old = """| Variant | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| :--- | :---: | :---: | :---: |
| LettuceDetect | 239 | 1.7029 | 0.7563 |
| full_verifier_filter | 657 | 2.7504 | 2.0791 |
| verifier_nli_filter | 670 | **4.0090** | 2.1203 |"""

t11_new = """| Variant | Sentence Ratings | Mean Sentence Rating | Sentence Coverage |
| :--- | :---: | :---: | :---: |
| LettuceDetect | 239 | 1.7029 | 0.7563 |
| full_verifier_filter | 657 | 2.7504 | 2.0791 |
| verifier_grounded_filter | 676 | 2.9053 | 2.1392 |
| verifier_intrinsic_filter | 688 | 2.8561 | 2.1772 |
| **verifier_nli_filter** | 670 | **4.0090** | 2.1203 |
| verifier_self_agreement_filter | 691 | 2.8379 | 2.1867 |"""

content = content.replace(t11_old, t11_new)

t12_old = """| Variant | Answer Ratings | Mean Answer Rating | Sent Ratings | Mean Sent Rating | Sent Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LettuceDetect | 316 | 0.9241 | 30 | 0.2000 | 0.0949 |
| full_verifier_filter | 316 | 0.5798 | 602 | 0.4186 | 1.9051 |
| verifier_nli_filter | 316 | **0.7701** | 612 | 0.7606 | 1.9367 |"""

t12_new = """| Variant | Answer Ratings | Mean Answer Rating | Sent Ratings | Mean Sent Rating | Sent Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LettuceDetect | 316 | 0.9241 | 30 | 0.2000 | 0.0949 |
| full_verifier_filter | 316 | 0.5798 | 602 | 0.4186 | 1.9051 |
| verifier_grounded_filter | 316 | 0.6062 | 624 | 0.4663 | 1.9747 |
| verifier_intrinsic_filter | 316 | 0.5833 | 645 | 0.4593 | 2.0411 |
| **verifier_nli_filter** | 316 | **0.7701** | 612 | 0.7606 | 1.9367 |
| verifier_self_agreement_filter | 316 | 0.5752 | 647 | 0.4490 | 2.0475 |"""

content = content.replace(t12_old, t12_new)

t13_old = """| Variant | Answer Ratings | Mean Answer Rating | Sent Ratings | Mean Sent Rating | Sent Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LettuceDetect | 316 | 0.9364 | 30 | 0.3304 | 0.0949 |
| full_verifier_filter | 316 | 0.7849 | 602 | 0.7153 | 1.9051 |
| verifier_nli_filter | 316 | **0.8358** | 612 | 0.8404 | 1.9367 |"""

t13_new = """| Variant | Answer Ratings | Mean Answer Rating | Sent Ratings | Mean Sent Rating | Sent Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LettuceDetect | 316 | 0.9364 | 30 | 0.3304 | 0.0949 |
| full_verifier_filter | 316 | 0.7849 | 602 | 0.7153 | 1.9051 |
| verifier_grounded_filter | 316 | 0.7788 | 624 | 0.7221 | 1.9747 |
| verifier_intrinsic_filter | 316 | 0.7688 | 645 | 0.7257 | 2.0411 |
| **verifier_nli_filter** | 316 | **0.8358** | 612 | 0.8404 | 1.9367 |
| verifier_self_agreement_filter | 316 | 0.7652 | 647 | 0.7233 | 2.0475 |"""

content = content.replace(t13_old, t13_new)

with open('docs/evaluation_result/Comprehensive_Final_Evaluation_Report.md', 'w') as f:
    f.write(content)

