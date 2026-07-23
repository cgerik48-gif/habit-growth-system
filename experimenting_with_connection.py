import gspread

# 1. Connect
gc = gspread.service_account(filename='imperial-ally-501823-t4-15c3b02c2e9e.json')
sheet = gc.open("Data")

for worksheet in sheet.worksheets():
    print(f"Tab Name: {worksheet.title} | Headers: {worksheet.row_values(1)}")