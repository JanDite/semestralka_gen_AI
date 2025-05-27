import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import plotly.graph_objects as go
from openai import OpenAI
from textwrap import wrap
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# Registrace fontu DejaVuSans
font_path = os.path.join("fonts", "DejaVuSans.ttf")
pdfmetrics.registerFont(TTFont("DejaVu", "/Users/jandite/Desktop/semestralka_gen_AI/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf"))

#načtení api klíče
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Převod jednotek
def mgdl_to_mmol(mgdl): return round(mgdl / 18.0, 1)

# Základní načtení dat
st.title("Přehled hladiny glukózy")
uploaded_file = st.file_uploader("📁 Nahraj CSV soubor s daty o glukóze", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Soubor úspěšně nahrán.")
else:
    df = pd.read_csv("/Users/jandite/Downloads/glucose_data.csv")
    st.info("ℹ️ Používají se výchozí data.")

# Úprava dat
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['value_mmol'] = df['value_mgdl'].apply(mgdl_to_mmol)

#graf
st.subheader("Interaktivní graf s vyznačenými hypo/hyper zónami glykemie.")
fig = go.Figure()

# Glukóza
fig.add_trace(go.Scatter(
    x=df['timestamp'],
    y=df['value_mmol'],
    mode='lines+markers',
    name='Glukóza (mmol/L)',
    line=dict(color='#33A5FF')
))

# Hypoglykémie zóna
fig.add_shape(
    type="rect",
    xref="x",
    yref="y",
    x0=df['timestamp'].min(),
    y0=0,
    x1=df['timestamp'].max(),
    y1=4.0,
    fillcolor="red",
    opacity=0.1,
    layer="below",
    line_width=0,
)

# Hyperglykémie zóna
fig.add_shape(
    type="rect",
    xref="x",
    yref="y",
    x0=df['timestamp'].min(),
    y0=10.0,
    x1=df['timestamp'].max(),
    y1=df['value_mmol'].max() + 1,
    fillcolor="yellow",
    opacity=0.1,
    layer="below",
    line_width=0,
)

# Layout
fig.update_layout(
    xaxis_title="Čas",
    yaxis_title="Glukóza (mmol/L)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📝 Záznam do deníku (volitelné)")
user_note = st.text_area(
    "Zapiš si, co by mohlo ovlivnit hladinu glukózy během dne (např. jídlo, aktivita, stres, spánek...)",
    placeholder="Např. K obědu jsem měl těstoviny, večer běhání, celý den byl stresující.",
    height=100
)

if user_note.strip():
    st.success("Záznam uložen do deníku.")
else:
    st.info("Záznam nebyl vyplněn – AI ho může ignorovat.")

# Statistiky ve dvou sloupcích
st.subheader("Statistiky")
col1, col2 = st.columns(2)
with col1:
    st.metric("📉 Minimum", f"{df['value_mmol'].min():.2f} mmol/L")
    st.metric("📈 Maximum", f"{df['value_mmol'].max():.2f} mmol/L")
with col2:
    st.metric("📊 Průměr", f"{df['value_mmol'].mean():.2f} mmol/L")

# Rozdělení podle rozsahu
st.subheader("Rozdělení podle cílového rozsahu")
in_range = df[(df['value_mmol'] >= 4.0) & (df['value_mmol'] <= 10.0)]
below = df[df['value_mmol'] < 4.0]
above = df[df['value_mmol'] > 10.0]

st.write(f"🟢 V cílovém rozsahu (4.0–10.0 mmol/L): **{len(in_range)} / {len(df)}** ({len(in_range)/len(df)*100:.1f} %)")
st.write(f"🔴 Hypoglykémie (<4.0 mmol/L): **{len(below)} / {len(df)}** ({len(below)/len(df)*100:.1f} %)")
st.write(f"🟡 Hyperglykémie (>10.0 mmol/L): **{len(above)} / {len(df)}** ({len(above)/len(df)*100:.1f} %)")

st.subheader("Kalkulačka sacharidového poměru")
insulin_units = st.number_input("Zadej celkovou denní dávku inzulínu (IU)", min_value=0.0, step=0.5)
carbs_grams = 350

if insulin_units > 0 and carbs_grams > 0:
    carb_ratio = carbs_grams / insulin_units
    st.session_state["insulin_units"] = insulin_units
    st.session_state["carbs_grams"] = carbs_grams

    st.success(f"Sacharidový poměr: 1 IU na {carb_ratio:.1f} g sacharidů")


def generate_ai_summary(df, user_note, below, above):
    # Získání času pro minimum a maximum
    min_row = df.loc[df['value_mmol'].idxmin()]
    max_row = df.loc[df['value_mmol'].idxmax()]

    # První hypo a hyper hodnoty (pokud existují)
    hypo_row = below.iloc[0] if not below.empty else None
    hyper_row = above.iloc[0] if not above.empty else None

    #získaní jednotek inzulínu a sacharidového poměru
    insulin_units = st.session_state.get("insulin_units", 0.0)
    carbs_grams = st.session_state.get("carbs_grams", 0.0)
    carb_ratio = carbs_grams / insulin_units if insulin_units > 0 else None

    prompt = f"""
Jsi zdravotní asistent pro diabetiky. Na základě dat a poznámky od uživatele vytvoř krátké shrnutí dat, upozorni na možné příčiny výkyvů a navrhni doporučení.
Pokud tam máš data o celkovém množství aplikovaného inzulínu (IU) a data o sacharidovém poměru, řekni také, zda odpovídá běžným hodnotám, nebo, zda
potřebuje upravit. Udělej i rychlé vyhodnocení proti půměrným hodnotám. Dle výzkumů by to mělo ideálně být v rozmezí 10 - 15 gramů na jednotku inzulínu.
V analýze ber také v potaz jídlo a události co mohou být zadáné v deníku.
Délku odpovědi měj maximálně 950 tokenů.

Data:
- Průměrná glukóza: {df['value_mmol'].mean():.2f} mmol/L
- Minimum: {min_row['value_mmol']:.2f} mmol/L v {min_row['timestamp'].strftime('%d.%m. %H:%M')}
- Maximum: {max_row['value_mmol']:.2f} mmol/L v {max_row['timestamp'].strftime('%d.%m. %H:%M')}
- Hypoglykémie: {len(below)} z {len(df)} měření
{"- První hypo: %.2f mmol/L v %s" % (hypo_row['value_mmol'], hypo_row['timestamp'].strftime('%d.%m. %H:%M')) if hypo_row is not None else ""}
- Hyperglykémie: {len(above)} z {len(df)} měření
{"- První hyper: %.2f mmol/L v %s" % (hyper_row['value_mmol'], hyper_row['timestamp'].strftime('%d.%m. %H:%M')) if hyper_row is not None else ""}

Dávka inzulínu:
- {insulin_units:.1f} IU

{"Sacharidový poměr: 1 IU na %.1f g sacharidů" % carb_ratio if carb_ratio else ""}

Poznámka uživatele:
{user_note if user_note.strip() else "žádná"}

Vytvoř shrnutí v češtině. Buď věcný a lidský.
"""

    response = client.chat.completions.create(
        model="gpt-4", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=1000
    )

    return response.choices[0].message.content


st.subheader("🤖 AI shrnutí")

if st.button("🧠 Vygenerovat shrnutí na základě dat"):
    with st.spinner("AI analyzuje data o glukóze..."):
        ai_summary = generate_ai_summary(df, user_note, below, above)
        st.success("✅ Shrnutí vygenerováno:")
        st.markdown(ai_summary)

# Funkce pro vytvoření PDF
def create_pdf_buffer(ai_summary=None):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    insulin_units = st.session_state.get("insulin_units", 0.0)
    carbs_grams = st.session_state.get("carbs_grams", 0.0)
    carb_ratio = carbs_grams / insulin_units if insulin_units > 0 else None
    #nastavení nadpisu
    c.setFont("DejaVu", 18)
    page_width = A4[0]
    c.drawCentredString(page_width / 2, 820, "Report o  glukóze")  # Y = výška stránky - posun

    text = c.beginText(40, 800)
    text.setFont("DejaVu", 12)
   # text.textLine("Report glukozy")
    text.textLine("")
    text.textLine(f"Prumerná glukóza: {df['value_mmol'].mean():.2f} mmol/L")
    text.textLine(f"Minimum: {df['value_mmol'].min():.2f} mmol/L")
    text.textLine(f"Maximum: {df['value_mmol'].max():.2f} mmol/L")
    text.textLine("")
    text.textLine(f"V cílovém rozmezí (4–10 mmol/L): {len(in_range)} / {len(df)} ({(len(in_range) / len(df)) * 100:.1f} %)")
    text.textLine(f"Hypoglykémie (<4.0 mmol/L): {len(below)} / {len(df)} ({(len(below) / len(df)) * 100:.1f} %)")
    text.textLine(f"Hyperglykémie (>10.0 mmol/L): {len(above)} / {len(df)} ({(len(above) / len(df)) * 100:.1f} %)")
   
    if insulin_units > 0:
        text.textLine("")
        text.textLine(f"Dávka inzulínu: {insulin_units:.1f} IU")
        if carb_ratio:
            text.textLine(f"Sacharidový poměr: 1 IU na {carb_ratio:.1f} g sacharidů")
   
    if user_note.strip():
        text.textLine("")
        text.textLine("Poznámka uživatele:")
        for line in user_note.strip().splitlines():
            text.textLine(f"• {line}")
    c.drawText(text)
    
    # 📈 Vytvoříme graf jako obrázek do paměti
    chart_buf = BytesIO()
    plt.figure(figsize=(6, 3))
    plt.plot(df['timestamp'], df['value_mmol'], label='Glukóza (mmol/L)', color='black')

    # Vyznačení zón
    plt.axhspan(0, 4.0, facecolor='blue', alpha=0.1, label='Hypoglykémie')
    plt.axhspan(10.0, df['value_mmol'].max() + 1, facecolor='red', alpha=0.1, label='Hyperglykémie')

    plt.xlabel('Čas')
    plt.ylabel('Glukóza (mmol/L)')
    plt.title('Graf glukózy v čase')
    #plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(chart_buf, format='PNG')
    plt.close()
    chart_buf.seek(0)

    # Přidáme obrázek do PDF
    image = ImageReader(chart_buf)
    c.drawImage(image, x=40, y=400, width=500, height=200)

    #AI shrnutí pod grafem
    if ai_summary and ai_summary.strip():
        lines = ai_summary.strip().splitlines()
        y_start = 300
        y_step = 14
        y = y_start
        max_width = 80

        # Nadpis
        c.setFont("DejaVu", 14)
        c.drawString(40, y, "AI shrnutí")
        y -= 2 * y_step
        c.setFont("DejaVu", 12)

        for line in lines:
            wrapped_lines = wrap(line, width=max_width)
            for wrapped_line in wrapped_lines:
                if y < 40:  # Pokud jsme moc dole, začni novou stránku
                    c.showPage()
                    y = 800
                    c.setFont("DejaVu", 12)

                c.drawString(40, y, wrapped_line)
                y -= y_step
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Export tlačítko
st.download_button(
    label="📄 Stáhnout PDF report",
    data=create_pdf_buffer(ai_summary if 'ai_summary' in locals() else None),
    file_name="glucose_report.pdf",
    mime="application/pdf"
)