import re
import json

list_of_file_names = open('..\\useless.txt', 'r')
input_dir = '..\\..\\data_set\\data\\crawl\\data\\'

unmatched_list = open('..\\unmatched_pages.txt', 'w')
matches = 0
processed_files = 0

site_issues_pattern = re.compile(' domain has been (chang|modifi|park|retir|suspend|transferr)ed|view the non-flash'
                                     ' version|site (has expired|is( currently)? under construction|is temporarily unavailable)')

for_sale = re.compile('(th(e|is)|business|(com|net|org))( domain( name)?| (web)?site)? (is|may be)( available)? for'
                          ' sale|(inquire about|buy|claim|purchase) this (domain( name)?|(web)?site)|domain name for sale')

for file_name in list_of_file_names:
    file_name = file_name.rstrip()  # remove whitespace ending
    company_data = json.load(open(input_dir+file_name, 'r', encoding='utf-8'))
    combined_content = ' '.join(company_data['content'])

    if site_issues_pattern.search(combined_content) is not None or for_sale.search(combined_content) is not None:
        print('Matching part found in: '+file_name+'\n')
        matches += 1
    else:
        print('Couldn\'t match '+file_name+'\n')
        unmatched_list.write(file_name+'\n')
    processed_files += 1
    print('Files processed: '+str(processed_files)+'\n')

print('Matched '+str(matches)+' files.\n')
