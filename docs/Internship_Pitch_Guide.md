# Research Internship Pitch Guide & Lab Outreach Blueprint

This guide provides targeted lab recommendations, pitch strategies, and cold email templates to showcase your completed Physics-Informed Digital Machining AI project to top international and national professors.

---

## 1. Top Research Labs & Professors to Target

### 🇺🇸 United States
1. **Purdue University** — *Smart Machine Tool / Digital Manufacturing Labs* (Prof. Tony Schmitz, Prof. Yung Shin)
2. **Georgia Institute of Technology** — *Machining Dynamics & Smart Manufacturing* (Prof. Thomas Kurfess, Prof. Christopher Saldana)
3. **University of Michigan, Ann Arbor** — *Mechatronics & Machining Systems* (Prof. Kira Barton, Prof. Chinedum Okwudire)
4. **Carnegie Mellon University (CMU)** — *Cyber-Physical Manufacturing AI & Robotics* (Prof. Rahul Mangharam, Prof. Kenji Shimada)
5. **MIT** — *Manufacturing and Productivity Lab (LMP)* (Prof. Brian Anthony, Prof. John Hart)

### 🇮🇳 India (IITs & IISc)
1. **IIT Madras** — *Manufacturing Engineering Section / Cyber-Physical Systems* (Prof. N. Ramesh Babu, Prof. M. S. Shunmugam)
2. **IIT Bombay** — *Machining Dynamics & Precision Engineering* (Prof. Ramesh Singh, Prof. S. S. Joshi)
3. **IIT Delhi** — *Smart Manufacturing & Automation Lab* (Prof. Sunil Jha, Prof. Pulak Mohan Pandey)
4. **IISc Bangalore** — *Department of Mechanical Engineering* (Prof. Ashitava Ghosal)

### 🇪🇺 Europe & UK
1. **TU Munich / RWTH Aachen (Germany)** — *Machine Tools and Production Engineering (WZL)*
2. **University of Sheffield (UK)** — *Advanced Manufacturing Research Centre (AMRC)*

---

## 2. High-Converting Cold Email Pitch Template

> **Subject**: Prospective Research Intern: Physics-Informed Real-Time Machining AI & Chatter Generalization
>
> Dear Professor [Last Name],
>
> I hope you are having a productive semester. I have been following your lab's recent work on [mention specific topic from their recent paper, e.g., intelligent machining dynamics, chatter suppression, or digital twins].
>
> Over the past few months, I built a **Physics-Informed Digital Twin framework for milling chatter detection and cross-tool stability generalization** using the open-source Tony Schmitz Digital Machining Database (**9,160 experimental cuts across 42 distinct dynamic setups**):
>
> 1. **Cross-Tool Generalization (GroupKFold / LODO)**: Rather than relying on naive train/test splits that leak tool dynamics, I evaluated models across mutually exclusive tool natural frequencies ($f_n$) and damping ratios, achieving **90.68% accuracy and 0.966 ROC-AUC on completely unseen tool configurations**.
> 2. **Real-Time Edge Latency Profiling**: Benchmarked end-to-end execution across 5,000 continuous 50 ms sliding windows ($10\text{ kHz}$ sampling) — achieving a median latency of **$7.76\text{ ms}$ (p95: $10.56\text{ ms}$)**, proving hard real-time viability for closed-loop CNC spindle speed override control.
> 3. **Interactive Digital Twin & Live Demo**: I deployed a full interactive Streamlit digital twin with Altintaş-Budak stability lobe overlays, 3D attractor orbits, and SHAP explainability.
>
> * **Interactive Demo & Code Repository**: [Link to your GitHub / Hosted Streamlit App]
> * **2-Page Extended Abstract**: Attached as a brief PDF summary for your quick review.
>
> I would be thrilled to contribute to your lab's ongoing research on [Lab Project Name] as a research intern during [Semester / Summer 2026]. Would you have 10 minutes for a brief introductory call to discuss potential research alignment?
>
> Thank you for your time and consideration.
>
> Sincerely,  
> **[Your Name]**  
> [Your University & Degree Program]  
> [LinkedIn Profile] | [GitHub Profile]

---

## 3. How to Ace the Research Interview

When a professor replies and invites you to a 10-minute call:

1. **Be Ready to Share Your Screen**: Open your **Streamlit dashboard (`app.py`)** and run a live demo. Show them how the star marker on the stability lobe turns red and triggers a chatter alarm in $0.63\text{ ms}$.
2. **Explain the GroupKFold Difference**: Highlight that standard ML papers in machining suffer from data leakage because they test on the same tool. Explain that you specifically tested on *unseen dynamic dataset groups*.
3. **Connect to Closed-Loop Control**: Explain that your $<8\text{ ms}$ latency fits comfortably within a $50\text{ ms}$ CNC buffer, allowing the controller to shift spindle RPM to a stable lobe before the tool breaks.
