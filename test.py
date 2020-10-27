# s = input("Amount of money (KRW):")

# try:
#     m = int(float(s) / 10) * 10
#     print("%s Won" % ("{:,}".format(m)))
# except ValueError:
#     print("Invalid value.")

bill_unit = [50000, 10000, 5000, 1000, 500, 100, 50, 10]
s = input("Amount of money (KRW):")
# try:
#     m = int(s)
#     if m > 0:
#         print(f'{m}원 화폐 매수 계산')
#         for i in bill_unit:
#             print(i)
#     #         mc = m//i
#     #         print(f'{i:6}원   : {mc:3}개')
#     #         m -= (i*mc)
#     else:
#         print('0 이상의 값을 입력 하세요.')
# except ValueError:
#     print("Invalid value.")

try:
    m = int(s)
    if m <= 0:
        print("유효한 값이 아닙니다.")
    else:        
        print(f"{m}원 화폐 매수 계산")
        print('=' * 30)
        total = 0
        for unit in bill_unit:
            number = m // unit
            unitsum = number * unit
            print(f"{unit:6} Won : {number:3}  {unitsum:8}")
            m -= unitsum
            total += unitsum
        
        print('=' * 30)
        print(f'Sum : {total}')

except ValueError:
    print("Invalid Value.")