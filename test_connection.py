import gspread

# The definitive way to connect in gspread 6.x
gc = gspread.service_account(filename='imperial-ally-501823-t4-15c3b02c2e9e.json')

# Open by exact name
sheet = gc.open("Data").sheet1

# Print the result
print("Connection successful! Header row found:")
print(sheet.row_values(1))