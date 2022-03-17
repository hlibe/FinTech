*=======================================================
*
*        Summary statistics
*
*=======================================================
* 

  global path "/Users/HaoLI/Stata/credit"    
  global D    "$path/data"      //范例数据
  global R    "$path/refs"      //参考文献
  global Out  "$path/out"       //结果：图形和表格
  adopath +   "$PP/adofiles"    //外部命令 
  cd "$D"
  
 
  //import delimited data1020use.csv, clear // data before this dofile
  import delimited data1210rename_use.csv, clear // data before this dofile
  //destring ttlamt_first8mon	ttlcnt_first8mon , replace force
  //use "data1020.dta", clear  // data processed after this dofile
 
 
sum2docx gender age edu_level_1 edu_level_2 edu_level_3 edu_level_5 edu_level_6 edu_level_7  housing_flag log_salary_level midnight_amt_mean  night_amt_mean  morning_amt_mean debit_amt_mean credit_amt_mean dcwt_amtpcnt95p dcwt_ordpmon99p dcmwth  cmwth_mcht_cnt ltry_ttlamt_ttlexpd realestate_mon_amtpexpd  dbldg vdsh_effmonratio0 water_effmonratio0 using 1.docx, replace title("表 1: 描述性统计")
shellout 1.docx

asdoc sum, append save(summary.doc) title(Summary statistics) stat(N mean sd min p25 p50 p75 max) fs(7) dec(3)
