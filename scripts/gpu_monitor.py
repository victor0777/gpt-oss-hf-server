#!/usr/bin/env python3
"""
Multi-GPU 부하 분산 모니터링
"""

import subprocess
import time
import json
from datetime import datetime

def monitor_gpu():
    """GPU 사용률 모니터링"""
    
    print("="*70)
    print("Multi-GPU 부하 분산 모니터링")
    print(f"시작: {datetime.now()}")
    print("="*70)
    
    monitoring_duration = 300  # 5분
    check_interval = 10  # 10초마다
    
    start_time = time.time()
    end_time = start_time + monitoring_duration
    
    checkpoints = []
    
    while time.time() < end_time:
        elapsed = time.time() - start_time
        
        # nvidia-smi로 GPU 상태 조회
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 6:
                    idx, name, mem_used, mem_total, util, temp = parts[:6]
                    gpu_info = {
                        'index': int(idx),
                        'name': name.strip(),
                        'memory_used_mb': float(mem_used),
                        'memory_total_mb': float(mem_total),
                        'memory_pct': float(mem_used) / float(mem_total) * 100,
                        'utilization': int(util),
                        'temperature': int(temp)
                    }
                    gpus.append(gpu_info)
            
            # 분석
            active_gpus = sum(1 for g in gpus if g['utilization'] > 10)
            total_util = sum(g['utilization'] for g in gpus)
            avg_util = total_util / len(gpus) if gpus else 0
            
            # 부하 균형 점수 계산 (0-1, 1이 완벽한 균형)
            if active_gpus > 0:
                util_values = [g['utilization'] for g in gpus if g['utilization'] > 10]
                if len(util_values) > 1:
                    avg = sum(util_values) / len(util_values)
                    variance = sum((x - avg) ** 2 for x in util_values) / len(util_values)
                    balance_score = 1 / (1 + variance / 100)  # 정규화
                else:
                    balance_score = 0.5  # 단일 GPU만 활성
            else:
                balance_score = 0
            
            checkpoint = {
                'elapsed_sec': round(elapsed),
                'timestamp': datetime.now().isoformat(),
                'gpus': gpus,
                'active_count': active_gpus,
                'avg_utilization': round(avg_util, 1),
                'balance_score': round(balance_score, 3)
            }
            checkpoints.append(checkpoint)
            
            # 출력
            print(f"\n[{elapsed:3.0f}초] GPU 상태")
            print(f"  활성 GPU: {active_gpus}/4")
            print(f"  평균 사용률: {avg_util:.1f}%")
            print(f"  부하 균형 점수: {balance_score:.3f}")
            
            for gpu in gpus:
                status = "🟢" if gpu['utilization'] > 10 else "⚪"
                print(f"  {status} GPU{gpu['index']}: "
                      f"사용률={gpu['utilization']:3d}% | "
                      f"메모리={gpu['memory_pct']:5.1f}% | "
                      f"온도={gpu['temperature']}°C")
            
            # 경고
            if active_gpus == 1 and elapsed > 60:
                print("  ⚠️ 경고: Multi-GPU 활용 미흡 (단일 GPU만 사용)")
            elif balance_score < 0.5 and active_gpus > 1:
                print("  ⚠️ 경고: GPU 간 부하 불균형")
            
        except Exception as e:
            print(f"  에러: {e}")
        
        time.sleep(check_interval)
    
    # 최종 분석
    print("\n" + "="*70)
    print("Multi-GPU 모니터링 완료")
    print("="*70)
    
    if checkpoints:
        # 평균 계산
        avg_active = sum(c['active_count'] for c in checkpoints) / len(checkpoints)
        avg_balance = sum(c['balance_score'] for c in checkpoints) / len(checkpoints)
        
        print(f"\n📊 요약:")
        print(f"  평균 활성 GPU: {avg_active:.1f}/4")
        print(f"  평균 부하 균형 점수: {avg_balance:.3f}")
        
        # 판정
        if avg_active >= 2 and avg_balance >= 0.6:
            print("  ✅ Multi-GPU 활용 양호")
        elif avg_active >= 2:
            print("  ⚠️ Multi-GPU 사용 중이나 부하 불균형")
        else:
            print("  ❌ Multi-GPU 활용 미흡")
    
    # 결과 저장
    with open("gpu_monitoring_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "duration_sec": monitoring_duration,
            "checkpoints": checkpoints,
            "summary": {
                "avg_active_gpus": avg_active if checkpoints else 0,
                "avg_balance_score": avg_balance if checkpoints else 0
            }
        }, f, indent=2)
    
    print("\n결과 저장: gpu_monitoring_results.json")

if __name__ == "__main__":
    monitor_gpu()