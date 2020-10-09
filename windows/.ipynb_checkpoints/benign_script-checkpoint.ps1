$a = Get-Content C:\temp\benign.txt

foreach( $i in $a) 
{
   try{
   copy $i e:\benign 
   }catch{
    "Skipped:" + $i
   }
}



$b = get-childitem e:\benign
foreach($i in $b){
 $hash = get-filehash $i.FullName
 $new = $i.name -replace $i.basename,$hash.hash 
 mv $i.FullName $new
  $i.FullName + "," + $new | out-file benign1.txt -append
}
