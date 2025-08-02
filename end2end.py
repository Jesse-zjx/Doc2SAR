import os
import csv
import io
import base64
import fitz
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
from openai import AzureOpenAI

from utils.prompts import END2END_PROMPT

API_CONFIG = {
    "api_base": "<your_azure_openai_endpoint>",
    "api_key": "<your_azure_openai_key>",
    "engine": "gpt-4o",
    "api_version": "2024-02-15-preview"
}


def extract_relevant_pages_to_images(pdf_path):
    image_streams = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(doc.page_count):  # 遍历所有页面
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image_stream = io.BytesIO(pix.tobytes("png"))
            image_streams.append((page_num, image_stream))    
    return image_streams


def call_model(query, image_streams, max_retries=0):

    client = AzureOpenAI(
        base_url=API_CONFIG["api_base"],
        api_key=API_CONFIG["api_key"],
        api_version=API_CONFIG["api_version"]
    )
    
    content_list = [{"type": "text","text": query}]
    for page_num, img_stream in image_streams:
        img_stream.seek(0)
        base64_image = base64.b64encode(img_stream.read()).decode('utf-8')
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
        
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=API_CONFIG["engine"],
                messages=[{"role": "user", "content": content_list}],
                temperature=0.0,
                timeout=150
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries:
                print(f"Retry {attempt}/{max_retries} failed. Retrying in 2s...")
                time.sleep(3)
            else:
                print(f"Failed to extract html after {max_retries} retries. Error: {e}")

    return ""


def parse_model_output(output):
    """解析模型输出的多CSV表格"""
    output = output.strip()
    if output == "NO_ACTIVITY_TABLES":
        return []
    
    
    if output.startswith("```csv"):
        output = output[len("```csv"):].strip()
    elif output.startswith("```"):
        output = output[len("```"):].strip()
    if output.endswith("```"):
        output = output[:-len("```")].strip()
        
    tables_str = output.split('---NEXT TABLE---')

    parsed_tables = []
    for table_str in tables_str:
        table_str = table_str.strip()
        if table_str: # Ensure table string is not empty
            parsed_tables.append(table_str)
            
    return parsed_tables
    

def merge_tables_by_header(tables):
    header_groups = defaultdict(list)
    
    for csv_str in tables:
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        
        if len(rows) < 2:
            continue
        
        header = tuple(rows[0])
        data_rows = rows[1:]
        header_groups[header].extend(data_rows)
    
    merged_tables = {}
    for header, data_rows in header_groups.items():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)
        for row in data_rows:
            writer.writerow(row)
        merged_tables[header] = output.getvalue()
    return merged_tables


def save_csv_tables(merged_tables, base_filename, output_dir):
    saved_files = []
    for i, (header, csv_data) in enumerate(merged_tables.items()):
        filename = f"{base_filename}_{i}.csv"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_data)
        saved_files.append(filepath)
    return saved_files


def process_pdf_file(pdf_path, output_dir):
    base_name = os.path.basename(pdf_path)[:-4]
    image_streams = extract_relevant_pages_to_images(pdf_path)
    model_output = call_model(END2END_PROMPT, image_streams)
    for _, img_stream in image_streams:
        img_stream.close()
    csv_tables = parse_model_output(model_output)
    merged_tables = merge_tables_by_header(csv_tables)
    return save_csv_tables(merged_tables, base_name, output_dir)


def process_pdf_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    all_saved_files = []
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(input_dir, pdf_file)
        saved_files = process_pdf_file(pdf_path, output_dir)
        all_saved_files.extend(saved_files)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', required=True, type=str)
    parser.add_argument('--pred', required=True, type=str)
    args = parser.parse_args()
    process_pdf_directory(args.dirs, args.pred)
